package io.rj93.sarcasm.cnn;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator.UnknownWordHandling;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
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
import org.json.JSONObject;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import io.rj93.sarcasm.filters.TestFileFilter;
import io.rj93.sarcasm.filters.TrainFileFilter;
import io.rj93.sarcasm.iterators.CnnSentenceChannelDataSetIterator;
import io.rj93.sarcasm.utils.DataHelper;
import io.rj93.sarcasm.utils.PrettyTime;

public class TextCNN {
	
	private static final Logger logger = LogManager.getLogger(TextCNN.class);
	private static final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
			
	private int nChannels;
	private String[] channelNames;
	private int nOutputs; // number of labels
	private int batchSize;
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
		for (Channel channel : channels){
			vectorSize += channel.getSize();
		}
		this.channels = channels;
		this.vectorSize = vectorSize;
		logger.info("Vector total size: " + vectorSize);
		
		this.seed = seed;
		this.conf = getConf();
		model = new ComputationGraph(conf);
        model.init();
        
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
	
	private ConvolutionLayer getConvolutionLayer(int noWords){
		return new ConvolutionLayer.Builder()
			.kernelSize(noWords,vectorSize)
			.stride(1,vectorSize)
			.nIn(nChannels)
			.nOut(cnnLayerFeatureMaps)
			.build();
	}
	
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
				
				String label;
				if (f.getAbsolutePath().contains("pos")){
					label = "positive";
					posCount += s.size();
				} else {
					label = "negative";
					negCount += s.size();
				}
				for (int i = 0; i < s.size(); i++){
					labels.add(label);
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		logger.info("No. positive: " + posCount + ", No. negative: " + negCount);
		LabeledSentenceProvider sentenceProvider = new CollectionLabeledSentenceProvider(sentences, labels, new Random(seed));
		
		MultiDataSetIterator iter = new CnnSentenceChannelDataSetIterator.Builder()
        		.sentenceProvider(sentenceProvider)
                .wordVectors(channels)
                .minibatchSize(batchSize)
                .build();
		
		return iter;
	}
	
	public void train(List<File> trainFiles, List<File> testFiles) throws IOException {
		
		logger.info("train - trainFiles: " + trainFiles.size() + ", testFiles: " + testFiles.size());
		
		MultiDataSetIterator trainIter = getMultiDataSetIterator(trainFiles);
		MultiDataSetIterator testIter = getMultiDataSetIterator(testFiles);
		labelsMap = ((CnnSentenceChannelDataSetIterator) testIter).getLabelsMap();
		
		logger.info("Training Model...");
		for (int i = 0; i < nEpochs; i++){
			
			
			logger.info("Starting epoch " + i + "... ");
			long start = System.nanoTime();
			model.fit(trainIter);
			long diff = System.nanoTime() - start;
			logger.info("Epoch " + i + " complete in " + PrettyTime.prettyNano(diff) + ". Starting evaluation...");
			
			start = System.nanoTime();
			ComputationGraph graph = (ComputationGraph) model;
            Evaluation evaluation = graph.evaluate(testIter);
            diff = System.nanoTime() - start;
            logger.info("Evaluation complete in: " + PrettyTime.prettyNano(diff));
            
            logger.info(evaluation.stats());
            
            trainIter.reset();
            testIter.reset();
		}
		logger.info("Training Complete");
		
		String dir = getModelSaveDir();
		save(dir, "model.bin");
	}
	
	public void trainEarlyStopping(List<File> trainFiles, List<File> testFiles) throws IOException {
		
		logger.info("trainEarlyStopping - trainFiles: " + trainFiles.size() + ", testFiles: " + testFiles.size());
		
		String dir = getModelSaveDir();
		new File(dir).mkdirs();
		
		MultiDataSetIterator trainIter = getMultiDataSetIterator(trainFiles);
		MultiDataSetIterator testIter = getMultiDataSetIterator(testFiles);
		labelsMap = ((CnnSentenceChannelDataSetIterator) testIter).getLabelsMap();
		
		EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
				.epochTerminationConditions(new MaxEpochsTerminationCondition(30))
				.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(1, TimeUnit.HOURS))
				.scoreCalculator(new DataSetLossCalculatorCG(testIter, true))
		        .evaluateEveryNEpochs(1)
				.modelSaver(new LocalFileGraphSaver(dir))
				.build();
		
		EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConf, model, trainIter, null);
		
		EarlyStoppingResult<ComputationGraph> result = trainer.fit();
		
		logger.info("Termination reason: " + result.getTerminationReason());
		logger.info("Termination details: " + result.getTerminationDetails());
		logger.info("Total epochs: " + result.getTotalEpochs());
		logger.info("Best epoch number: " + result.getBestModelEpoch());
		logger.info("Score at best epoch: " + result.getBestModelScore());
		
		ComputationGraph bestModel = result.getBestModel();
		testIter.reset();
		
		logger.info("Starting evaluation...");
		long start = System.nanoTime();
        Evaluation evaluation = bestModel.evaluate(testIter);
        long diff = System.nanoTime() - start;
        logger.info("Evaluation complete in: " + PrettyTime.prettyNano(diff));
        
        logger.info(evaluation.stats());
        
        save(dir, "model.bin");
	}
	
	private static String getModelSaveDir(){
		String dir = DataHelper.MODELS_DIR + dateFormat.format(System.currentTimeMillis()) + "/";
		return dir;
	}
	
	public void save(String dir, String fileName) throws IOException{
		save(dir, fileName, false);
	}
	
	public void save(String dir, String fileName, boolean saveUpdater) throws IOException{	
		new File(dir).mkdirs();
		String file = FilenameUtils.concat(dir, fileName);
		ModelSerializer.writeModel(model, file, saveUpdater);
		writeConfig(dir);
	}
	
	public void writeConfig(String dir) throws IOException{
		String file = FilenameUtils.concat(dir, "config.json");
		
		JSONObject labels = new JSONObject();
		for (Entry<String, Integer> entry : labelsMap.entrySet()){
			labels.put(entry.getKey(), entry.getValue());
		}
		
		JSONObject channels = new JSONObject();
		for (int channel = 0; channel < nChannels; channel++){
			channels.append(String.valueOf(channel), this.channels.get(channel).toString());
		}
		
		JSONObject configJson = new JSONObject();
		configJson.put("labels", labels);
		configJson.put("channels", channels);
		PrintWriter writer = new PrintWriter(file, "UTF-8");
		writer.println(configJson.toString(4));
		writer.close();
	}
	
	public Prediction predict(String sentence){
		
		INDArray[] features = new INDArray[nChannels];
		for (int channel = 0; channel < nChannels; channel++){
			features[channel] = channels.get(channel).getFeatureVector(sentence);
		}
		INDArray result = model.outputSingle(features);
		
		return new Prediction(result.getDouble(labelsMap.get("positive")), result.getDouble(labelsMap.get("negative") ));
		
	}
	
	public void test(List<File> testFiles) throws FileNotFoundException {
	}
	
	public void startUIServer(){
		logger.info("Starting UI Server");
		uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage(); 
		uiServer.attach(statsStorage);
		model.setListeners(new StatsListener(statsStorage));
	}
	
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
//		channels.add(new WordVectorChannel(DataHelper.GOOGLE_NEWS_WORD2VEC, true, UnknownWordHandling.UseUnknownVector, maxSentenceLength));
		channels.add(new WordVectorChannel(DataHelper.WORD2VEC_DIR + "all-preprocessed-300-test.emb", true, UnknownWordHandling.UseUnknownVector, maxSentenceLength));

		List<File> trainFiles = getSarcasmFiles(true);
		List<File> testFiles = getSarcasmFiles(false);
		
		try {
			TextCNN cnn = new TextCNN(outputs, batchSize, epochs, channels);
			cnn.startUIServer();
			long start = System.currentTimeMillis();
			cnn.train(trainFiles, testFiles);
			long diff = System.currentTimeMillis() - start;
			logger.info("Time taken: " + PrettyTime.prettyNano(diff));
		} catch (Exception e){
			e.printStackTrace();
		} finally {
			System.exit(-1);
		}
		
	}
	
	private static List<File> getSentimentFiles(boolean training) throws FileNotFoundException{
		logger.info("using sentiment files");
		String dirStr = "C:/Users/Richard/AppData/Local/Temp/dl4j_w2vSentiment/aclImdb/";
		File[] filesPos;
		File[] filesNeg;
		if (training) {
			filesPos = new File(dirStr + "train/pos").listFiles();
			filesNeg = new File(dirStr + "train/neg").listFiles();
		} else {
			filesPos = new File(dirStr + "test/pos").listFiles();
			filesNeg = new File(dirStr + "test/neg").listFiles();
		}
		
		List<File> files = new ArrayList<File>();
		for (int i = 0; i < filesPos.length; i++){
			files.add(filesPos[i]);
		}
		for (int i = 0; i < filesNeg.length; i++){
			files.add(filesNeg[i]);
		}
		return files;
	}
	
	
	private static List<File> getSarcasmFiles(boolean training) throws FileNotFoundException{
		logger.info("using sarcastic files");
		File dir = new File(DataHelper.PREPROCESSED_DATA_DIR + "2015-quick");
		if (training)
			return DataHelper.getFilesFromDir(dir, new TrainFileFilter(2), true);
		else 
			return DataHelper.getFilesFromDir(dir, new TestFileFilter(2), true);
	}

}
