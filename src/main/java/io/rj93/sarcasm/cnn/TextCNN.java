package io.rj93.sarcasm.cnn;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator.UnknownWordHandling;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.json.JSONArray;
import org.json.JSONObject;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import io.rj93.sarcasm.cnn.channels.Channel;
import io.rj93.sarcasm.cnn.channels.WordVectorChannel;
import io.rj93.sarcasm.iterators.ChannelDataSetIterator;
import io.rj93.sarcasm.preprocessing.TextPreProcessor;
import io.rj93.sarcasm.utils.DataHelper;
import io.rj93.sarcasm.utils.PrettyTime;

public class TextCNN {
	
	private static final Logger logger = LogManager.getLogger(TextCNN.class);
	private static final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
			
	private int nChannels;
	private String[] channelNames;
	private int nOutputs; // number of labels
	private int batchSize = 32;
	private int nEpochs;
	private int iterations = 1;
	private int seed;
	private List<Channel> channels;
	private int vectorSize;
	private int cnnLayerFeatureMaps = 100;
	private ComputationGraphConfiguration conf;
	private ComputationGraph model;
	private UIServer uiServer = null;
	private Map<String, Integer> labelsMap;
	
	public TextCNN(int nOutputs, int batchSize, int nEpochs, Channel channel){
		this(nOutputs, batchSize, nEpochs, Arrays.asList(channel), 12345);
	}
	
	public TextCNN(int nOutputs, int batchSize, int nEpochs, List<Channel> channels){
		this(nOutputs, batchSize, nEpochs, channels, 12345);
	}
	
	public TextCNN(int nOutputs, int batchSize, int nEpochs, List<Channel> channels, int seed){
		this.nChannels = channels.size();
		channelNames = new String[nChannels];
		for (int i = 0; i < nChannels; i++){
			channelNames[i] = "input" + i;
		}
		this.nOutputs = nOutputs;
		
		this.batchSize = batchSize;
		this.nEpochs = nEpochs;
		
		int vectorSize = 0;
		// sum total size of all the channels
		for (Channel channel : channels){
			vectorSize += channel.getSize();
		}
		this.channels = channels;
		this.vectorSize = vectorSize;
		logger.info("Vector total size: " + vectorSize);
		
		this.seed = seed;
		this.conf = getConf();
		this.model = new ComputationGraph(conf);
        this.model.init();
	}

	public TextCNN(ComputationGraph model, List<Channel> channels, Map<String, Integer> labels, int seed) {
		this.model = model;
		this.channels = channels;
		this.nChannels = channels.size();
		this.labelsMap = labels;
		this.seed = seed;
	}

	private ComputationGraphConfiguration getConf(){
		
		ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
				.weightInit(WeightInit.RELU)
				.activation(Activation.LEAKYRELU)
				.updater(Updater.ADAM)
				.convolutionMode(ConvolutionMode.Same)
				.seed(seed)
				.iterations(iterations)
				.regularization(true).l2(0.0001)
				.learningRate(0.01)
				.graphBuilder()
				.addInputs(channelNames)
				.addLayer("cnn3", getConvolutionLayer(3), channelNames)
				.addLayer("cnn4", getConvolutionLayer(4), channelNames)
				.addLayer("cnn5", getConvolutionLayer(5), channelNames)
				.addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")
				.addLayer("globalPool", new GlobalPoolingLayer.Builder()
						.poolingType(PoolingType.MAX)
						.build(), "merge")
				.addLayer("out", new OutputLayer.Builder()
						.lossFunction(LossFunctions.LossFunction.MCXENT)
						.activation(Activation.SOFTMAX)
						.nIn(3*cnnLayerFeatureMaps)
						.nOut(nOutputs)
						.build(), "globalPool")
				.setOutputs("out")
				.build();
		
		return conf;
	}
	
	/**
	 * Create a new convolutional layer, with the specificed window of words
	 * @param noWords
	 * @return
	 */
	private ConvolutionLayer getConvolutionLayer(int noWords){
		return new ConvolutionLayer.Builder()
			.kernelSize(noWords,vectorSize)
			.stride(1,vectorSize)
			.nIn(nChannels)
			.nOut(cnnLayerFeatureMaps)
			.build();
	}
	
	/**
	 * Creates the DataSetIterator form the map
	 * @param map map containing texts and labels
	 * @return
	 */
	private MultiDataSetIterator getMultiDataSetIterator(Map<String, String> map){
		List<String> sentences = new ArrayList<String>();
		List<String> labels = new ArrayList<String>();
		for (Entry<String, String> entry : map.entrySet()){
			sentences.add(entry.getKey());
			labels.add(entry.getValue());
		}
		
		return getMultiDataSetIterator(sentences, labels);
	}
	
	/**
	 * Reads each file, and labels the texts accordingly to create the DataSetIterator
	 * @param files the files to be read
	 * @return
	 * @throws FileNotFoundException
	 */
	private MultiDataSetIterator getMultiDataSetIterator(List<File> files) throws FileNotFoundException {
		
		List<String> sentences = new ArrayList<String>();
		List<String> labels = new ArrayList<String>();
		int posCount = 0;
		int negCount = 0;
		for (File f : files){
			
			List<String> s;
			try {
				s = FileUtils.readLines(f, "UTF-8");
				sentences.addAll(s);
				
				// determine correct label by looking at file path
				String label;
				if (f.getAbsolutePath().contains("pos")){
					label = "positive";
					posCount += s.size();
				} else {
					label = "negative";
					negCount += s.size();
				}
				
				// insert correct label
				for (int i = 0; i < s.size(); i++){
					labels.add(label);
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		logger.info("No. positive: " + posCount + ", No. negative: " + negCount);
		return getMultiDataSetIterator(sentences, labels);
	}
	
	/**
	 * instantiates a ChannelDataSetIterator using the sentences and labels
	 * @param sentences
	 * @param labels
	 * @return
	 */
	private MultiDataSetIterator getMultiDataSetIterator(List<String> sentences, List<String> labels){
		LabeledSentenceProvider sentenceProvider = new CollectionLabeledSentenceProvider(sentences, labels, new Random(seed));
		
		MultiDataSetIterator iter = new ChannelDataSetIterator.Builder()
        		.sentenceProvider(sentenceProvider)
                .wordVectors(channels)
                .minibatchSize(batchSize)
                .build();
		
		return iter;
	}
	
	/**
	 * trains the model on the training and testing maps
	 * @param trainMap sentences and labels of the training dataset
	 * @param testMap sentences and labels of the testing dataset
	 * @throws IOException
	 */
	public void train(Map<String, String> trainMap, Map<String, String> testMap) throws IOException{
		
		logger.info("train - trainMap: " + trainMap.size() + ", testMap: " + testMap.size());
		
		MultiDataSetIterator trainIter = getMultiDataSetIterator(trainMap);
		MultiDataSetIterator testIter = getMultiDataSetIterator(testMap);
		
		train(trainIter, testIter);
	}
	
	/**
	 * trains the model on the training and testing files
	 * @param trainFiles training files to be read
	 * @param testFiles testing files to be read
	 * @throws IOException
	 */
	public void train(List<File> trainFiles, List<File> testFiles) throws IOException {
		
		logger.info("train - trainFiles: " + trainFiles.size() + ", testFiles: " + testFiles.size());
		
		MultiDataSetIterator trainIter = getMultiDataSetIterator(trainFiles);
		MultiDataSetIterator testIter = getMultiDataSetIterator(testFiles);
		
		train(trainIter, testIter);
	}
	
	/**
	 * trains the model on the training and testing iterators
	 * @param trainIter training iterator
	 * @param testIter testing iterator
	 * @throws IOException
	 */
	private void train(MultiDataSetIterator trainIter, MultiDataSetIterator testIter) throws IOException{
		labelsMap = ((ChannelDataSetIterator) trainIter).getLabelsMap();
		
		logger.info("Training Model...");
		for (int i = 0; i < nEpochs; i++){
			
			logger.info("Starting epoch " + i + "... ");
			long start = System.nanoTime();
			model.fit(trainIter);
			long diff = System.nanoTime() - start;
			logger.info("Epoch " + i + " complete in " + PrettyTime.prettyNano(diff) + ". Starting evaluation...");
			
			Evaluation evaluation = test(testIter);            
            logger.info(evaluation.stats());
            
            trainIter.reset();
		}
		logger.info("Training Complete");
		
		String dir = getModelSaveDir();
		save(dir, "model.bin");
	}
	
	/**
	 * trains the model on the training and testing maps, using the provided early stopping config
	 * @param trainMap sentences and labels of the training dataset
	 * @param testMap sentences and labels of the testing dataset
	 * @param esConf the early stopping configuration
	 * @return the EarlyStoppingResult
	 * @throws IOException
	 */
	public EarlyStoppingResult<ComputationGraph> train(Map<String, String> trainMap, Map<String, String> testMap, EarlyStoppingConfiguration<ComputationGraph> esConf) throws IOException {
		String dir = getModelSaveDir();
		new File(dir).mkdirs();
		
		MultiDataSetIterator trainIter = getMultiDataSetIterator(trainMap);
		MultiDataSetIterator testIter = getMultiDataSetIterator(testMap);

		return train(trainIter, testIter, esConf);
	}
	
	/**
	 * trains the model on the training and testing files, using the provided early stopping config
	 * @param trainFiles training files to be read
	 * @param testFiles testing files to be read
	 * @param esConf the early stopping configuration
	 * @return the EarlyStoppingResult
	 * @throws IOException
	 */
	public EarlyStoppingResult<ComputationGraph> train(List<File> trainFiles, List<File> testFiles, EarlyStoppingConfiguration<ComputationGraph> esConf) throws IOException {
		String dir = getModelSaveDir();
		new File(dir).mkdirs();
		
		MultiDataSetIterator trainIter = getMultiDataSetIterator(trainFiles);
		MultiDataSetIterator testIter = getMultiDataSetIterator(testFiles);

		return train(trainIter, testIter, esConf);
	}
	
	/**
	 * trains the model on the training and testing iterators, using the provided early stopping config
	 * @param trainIter training iterator
	 * @param testIter testing iterator
	 * @param esConf the early stopping configuration
	 * @return the EarlyStoppingResult
	 * @throws IOException
	 */
	private EarlyStoppingResult<ComputationGraph> train(MultiDataSetIterator trainIter, MultiDataSetIterator testIter, EarlyStoppingConfiguration<ComputationGraph> esConf) throws IOException {
		
		String dir = getModelSaveDir();
		new File(dir).mkdirs();
		
		labelsMap = ((ChannelDataSetIterator) trainIter).getLabelsMap();
		
		esConf.setScoreCalculator(new DataSetLossCalculatorCG(testIter, true));
		esConf.setModelSaver(new LocalFileGraphSaver(dir));
		EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConf, model, trainIter, null);
		EarlyStoppingResult<ComputationGraph> result = trainer.fit();
        
		model = result.getBestModel();
		save(dir, "model.bin");
        return result;
	}
	
	/**
	 * Gets the model save dir, which uses the current system time to avoid over writting
	 * @return the model save dir
	 */
	private static String getModelSaveDir(){
		String dir = DataHelper.MODELS_DIR + dateFormat.format(System.currentTimeMillis()) + "/";
		return dir;
	}
	
	/**
	 * saves the model and configuration
	 * @param dir the dir to save
	 * @param fileName the name of the model
	 * @throws IOException
	 */
	public void save(String dir, String fileName) throws IOException{
		save(dir, fileName, false);
	}
	
	/**
	 * Saves the model and configuration
	 * @param dir the dir to save
	 * @param fileName the name of the model
	 * @param saveUpdater true if the model is going to be trained after being loaded
	 * @throws IOException
	 */
	public void save(String dir, String fileName, boolean saveUpdater) throws IOException{	
		new File(dir).mkdirs();
		String file = FilenameUtils.concat(dir, fileName);
		ModelSerializer.writeModel(model, file, saveUpdater);
		writeConfig(dir);
	}
	
	/**
	 * Writes the config of the cnn to a json file
	 * @param dir directory to write to
	 * @throws IOException
	 */
	public void writeConfig(String dir) throws IOException{
		String file = FilenameUtils.concat(dir, "config.json");
		
		JSONObject labels = new JSONObject();
		for (Entry<String, Integer> entry : labelsMap.entrySet()){
			labels.put(entry.getKey(), entry.getValue());
		}
		
		JSONArray channels = new JSONArray();
		for (int channel = 0; channel < nChannels; channel++){
			channels.put(this.channels.get(channel).getConfig());
		}
		
		JSONObject configJson = new JSONObject();
		configJson.put("labels", labels);
		configJson.put("channels", channels);
		configJson.put("seed", seed);
		PrintWriter writer = new PrintWriter(file, "UTF-8");
		writer.println(configJson.toString(4));
		writer.close();
	}
	
	/**
	 * Loads the model and configuration
	 * @param dir the dir to laod from
	 * @param fileName name of the model
	 * @return a TextCNN object
	 * @throws IOException
	 */
	@SuppressWarnings("rawtypes")
	public static TextCNN loadFromDir(String dir, String fileName) throws IOException{
			
		JSONObject config = loadConfig(dir);
		int seed = config.getInt("seed");
		
		List<Channel> channels = new ArrayList<Channel>();
		JSONArray channelsArray = config.getJSONArray("channels");
		for (int i = 0; i < channelsArray.length(); i++){
			channels.add(Channel.loadFromConfig(channelsArray.getJSONObject(i)));
		}
		
		Map<String, Integer> labels = new HashMap<String, Integer>();
		JSONObject labelObj = config.getJSONObject("labels");
		Iterator keysIter = labelObj.keys();
		while(keysIter.hasNext()){
			String key = (String) keysIter.next();
			int value = labelObj.getInt(key);
			labels.put(key, value);
		}
		
		String modelPath = FilenameUtils.concat(dir, fileName);
		ComputationGraph model = ModelSerializer.restoreComputationGraph(modelPath);
		
		return new TextCNN(model, channels, labels, seed);
	}
	
	/**
	 * reads the config file
	 * @param dir the dir to read from
	 * @return the JSON congifuration
	 * @throws IOException
	 */
	public static JSONObject loadConfig(String dir) throws IOException{
		String confFilePath = FilenameUtils.concat(dir, "config.json");
		
		List<String> lines = FileUtils.readLines(new File(confFilePath));
		StringBuilder sb = new StringBuilder();
		for (String line : lines){
			sb.append(line);
		}
		
		return new JSONObject(sb.toString());
	}
	
	/**
	 * predicts if a text is sarcastic or not
	 * @param sentence the text
	 * @return the prediction
	 */
	public Prediction predict(String sentence){
		TextPreProcessor preprocessor = new TextPreProcessor(false, false);
		String preProcessed = preprocessor.preProcess(sentence);
		
		INDArray[] features = new INDArray[nChannels];
		for (int channel = 0; channel < nChannels; channel++){
			features[channel] = channels.get(channel).getFeatureVector(preProcessed);
		}
		INDArray result = model.outputSingle(features);
		
		return new Prediction(result.getDouble(labelsMap.get("positive")), result.getDouble(labelsMap.get("negative") ));
		
	}
	
	/**
	 * evaluates the model on the test files
	 * @param testFiles the test files to be read
	 * @return the Evaluation
	 * @throws FileNotFoundException
	 */
	public Evaluation test(List<File> testFiles) throws FileNotFoundException {
		MultiDataSetIterator testIter = getMultiDataSetIterator(testFiles);
		
		return test(testIter);
	}
	
	/**
	 * evaluates the model on the test iterator
	 * @param testIter the test iterator
	 * @return the Evaluation
	 * @throws FileNotFoundException
	 */
	private Evaluation test(MultiDataSetIterator testIter){
		logger.info("Starting evaluation...");
		long start = System.nanoTime();
		ComputationGraph graph = (ComputationGraph) model;
        Evaluation evaluation = graph.evaluate(testIter);
        long diff = System.nanoTime() - start;
        logger.info("Evaluation complete in: " + PrettyTime.prettyNano(diff));
        
        return evaluation;
	}
	
	/**
	 * Starts the UI server to review training progress at localhost:9000
	 */
	public void startUIServer(){
		logger.info("Starting UI Server");
		uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage(); 
		uiServer.attach(statsStorage);
		model.setListeners(new StatsListener(statsStorage));
	}
	
	/**
	 * Stops the UI Server
	 */
	public void stopUIServer(){
		if  (uiServer != null)
			uiServer.stop();
	}
	
	public static void main(String[] args) throws IOException {
		
		int outputs = 2; 
		int batchSize = 32;
		int epochs = 1;
		int maxSentenceLength = 100;
	
		List<Channel> channels = new ArrayList<Channel>();
		channels.add(new WordVectorChannel(DataHelper.GOOGLE_NEWS_WORD2VEC, true, UnknownWordHandling.UseUnknownVector, maxSentenceLength));
		channels.add(new WordVectorChannel(DataHelper.WORD2VEC_DIR + "all-preprocessed-300.emb", true, UnknownWordHandling.UseUnknownVector, maxSentenceLength));
		channels.add(new WordVectorChannel(DataHelper.GLOVE, true, UnknownWordHandling.UseUnknownVector, maxSentenceLength));
		
		List<File> trainFiles = DataHelper.getSarcasmFiles(true, false);
		List<File> testFiles = DataHelper.getSarcasmFiles(false, false);
		
		try {
//			TextCNN cnn = new TextCNN(outputs, batchSize, epochs, channels);
//			cnn.startUIServer();
//			long start = System.nanoTime();
//			cnn.train(trainFiles, testFiles);
//			long diff = System.nanoTime() - start;
//			logger.info("Total time taken: " + PrettyTime.prettyNano(diff));
//			TextCNN cnn = TextCNN.loadFromDir(DataHelper.MODELS_DIR, "model.bin");
//			cnn.test(testFiles);
		} catch (Exception e){
			e.printStackTrace();
		} finally {
			System.exit(-1);
		}
		
	}

}
