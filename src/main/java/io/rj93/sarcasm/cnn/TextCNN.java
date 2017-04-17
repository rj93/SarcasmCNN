package io.rj93.sarcasm.cnn;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.FileUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculatorCG;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator.UnknownWordHandling;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
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
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.format.datetime.standard.DateTimeContextHolder;

import io.rj93.sarcasm.filters.TestFileFilter;
import io.rj93.sarcasm.filters.TrainFileFilter;
import io.rj93.sarcasm.iterators.CnnSentenceMultiDataSetIterator;
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
	private List<WordVectors> embeddings;
	private int vectorSize;
	private int maxSentenceLength;
	private int cnnLayerFeatureMaps = 100;
	private ComputationGraphConfiguration conf;
	private ComputationGraph model;
	private UIServer uiServer = null;
	private CnnSentenceDataSetIterator singleChannelIter;
	private CnnSentenceMultiDataSetIterator multiChannelIter;
	
	public TextCNN(int nOutputs, int batchSize, int nEpochs, WordVectors embedding, int maxSentenceLength){
		this(nOutputs, batchSize, nEpochs, Arrays.asList(embedding), maxSentenceLength, 12345);
	}
	
	public TextCNN(int nOutputs, int batchSize, int nEpochs, List<WordVectors> embeddings, int maxSentenceLength){
		this(nOutputs, batchSize, nEpochs, embeddings, maxSentenceLength, 12345);
	}
	
	public TextCNN(int nOutputs, int batchSize, int nEpochs, List<WordVectors> embeddings, int maxSentenceLength, int seed){
		this.nChannels = embeddings.size();
		channelNames = new String[nChannels];
		for (int i = 0; i < nChannels; i++){
			channelNames[i] = "input" + i;
		}
		this.nOutputs = nOutputs;
		
		this.batchSize = batchSize;
		this.nEpochs = nEpochs;
		
		int vectorSize = 0;
		for (WordVectors embedding : embeddings){
			if (embedding.getUNK() == null)
				embedding.setUNK("UNK");
			vectorSize += embedding.getWordVector(embedding.vocab().wordAtIndex(0)).length;
		}
		this.embeddings = embeddings;
		this.vectorSize = vectorSize;
		logger.info("Vector total size: " + vectorSize);
		
		this.maxSentenceLength = maxSentenceLength;
		this.seed = seed;
		this.conf = getConf();
		model = new ComputationGraph(conf);
        model.init();
        
//        if (nChannels == 1){
//        	singleChannelIter = new CnnSentenceDataSetIterator.Builder()
//                    .wordVectors(embeddings.get(0))
//                    .maxSentenceLength(maxSentenceLength)
//                    .useNormalizedWordVectors(false)
//                    .unknownWordHandling(UnknownWordHandling.UseUnknownVector)
//                    .build();
//        } else {
//        	multiChannelIter = new CnnSentenceMultiDataSetIterator.Builder()
//        			.wordVectors(embeddings)
//        			.maxSentenceLength(maxSentenceLength)
//        			.useNormalizedWordVectors(false)
//        			.unknownWordHandling(UnknownWordHandling.UseUnknownVector.UseUnknownVector)
//        			.build();
//        }
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
				.addLayer("cnn3", new ConvolutionLayer.Builder()
						.kernelSize(3,vectorSize)
						.stride(1,vectorSize)
						.nIn(nChannels)
						.nOut(cnnLayerFeatureMaps)
						.build(), channelNames)
				.addLayer("cnn4", new ConvolutionLayer.Builder()
						.kernelSize(4,vectorSize)
						.stride(1,vectorSize)
						.nIn(nChannels)
						.nOut(cnnLayerFeatureMaps)
						.build(), channelNames)
				.addLayer("cnn5", new ConvolutionLayer.Builder()
						.kernelSize(5,vectorSize)
						.stride(1,vectorSize)
						.nIn(nChannels)
						.nOut(cnnLayerFeatureMaps)
						.build(), channelNames)
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
	
	private DataSetIterator getDataSetIterator(List<File> files) throws FileNotFoundException{
		
		LabeledSentenceProvider sentenceProvider;
		if (files.size() < 1000){
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
			sentenceProvider = new CollectionLabeledSentenceProvider(sentences, labels, new Random(seed));
		} else {
			
			List<File> posFiles = new ArrayList<>();
			List<File> negFiles = new ArrayList<>();
			
			for (File f : files){
				if (f.getAbsolutePath().contains("pos")){
					posFiles.add(f);
				} else {
					negFiles.add(f);
				}
			}
			logger.info("No. positive: " + posFiles.size() + ", No. negative: " + negFiles.size());
			
			Map<String,List<File>> map = new HashMap<>();
			map.put("positive", posFiles);
			map.put("negative", negFiles);
			
			sentenceProvider = new FileLabeledSentenceProvider(map, new Random(seed));
		}
		
		
		DataSetIterator iter = new CnnSentenceDataSetIterator.Builder()
        		.sentenceProvider(sentenceProvider)
                .wordVectors(embeddings.get(0))
                .minibatchSize(batchSize)
                .maxSentenceLength(maxSentenceLength)
                .useNormalizedWordVectors(false)
                .unknownWordHandling(UnknownWordHandling.UseUnknownVector)
                .build();
        return iter;
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
		
		MultiDataSetIterator iter = new CnnSentenceMultiDataSetIterator.Builder()
        		.sentenceProvider(sentenceProvider)
                .wordVectors(embeddings)
                .minibatchSize(batchSize)
                .maxSentenceLength(maxSentenceLength)
                .useNormalizedWordVectors(false)
                .unknownWordHandling(UnknownWordHandling.UseUnknownVector)
                .build();
		
		return iter;
	}
	
	public void train(List<File> trainFiles, List<File> testFiles) throws IOException {
		
		logger.info("train - trainFiles: " + trainFiles.size() + ", testFiles: " + testFiles.size());
		
		if (embeddings.size() == 1){
			trainSingleChannel(trainFiles, testFiles);
		} else {
			trainMultiChannel(trainFiles, testFiles);
		}
	}
	
	private void trainSingleChannel(List<File> trainFiles, List<File> testFiles) throws IOException {
		
		logger.info("trainSingleChannel");
		
		DataSetIterator trainIter = getDataSetIterator(trainFiles);
		DataSetIterator testIter = getDataSetIterator(testFiles);
		
		logger.info("Training Model...");
		for (int i = 0; i < nEpochs; i++){
			
			
			logger.info("Starting epoch " + i + "... ");
			long start = System.nanoTime();
			model.fit(trainIter);
			long diff = System.nanoTime() - start;
			logger.info("Epoch " + i + " complete in " + PrettyTime.prettyNano(diff) + ". Starting evaluation...");
			
			start = System.nanoTime();
            Evaluation evaluation = model.evaluate(testIter);
            diff = System.nanoTime() - start;
            logger.info("Evaluation complete in: " + PrettyTime.prettyNano(diff));
            
            logger.info(evaluation.stats());
            
            trainIter.reset();
            testIter.reset();
		}
		logger.info("Training Complete");
	}
	
	private void trainMultiChannel(List<File> trainFiles, List<File> testFiles) throws IOException {
		
		logger.info("trainMultiChannel");
		
		MultiDataSetIterator trainIter = getMultiDataSetIterator(trainFiles);
		MultiDataSetIterator testIter = getMultiDataSetIterator(testFiles);
		
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
	}
	
	public void trainEarlyStopping(List<File> trainFiles, List<File> testFiles) throws IOException {
		
		logger.info("trainTest - trainFiles: " + trainFiles.size() + ", testFiles: " + testFiles.size());
		
		String dir = getModelDir();
		new File(dir).mkdirs();
		
		DataSetIterator trainIter = getDataSetIterator(trainFiles);
		DataSetIterator testIter = getDataSetIterator(testFiles);
		
		EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
				.epochTerminationConditions(new MaxEpochsTerminationCondition(30))
				.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(30, TimeUnit.MINUTES))
				.scoreCalculator(new DataSetLossCalculatorCG(testIter, true))
		        .evaluateEveryNEpochs(1)
				.modelSaver(new LocalFileGraphSaver(dir))
				.build();

		EarlyStoppingGraphTrainer trainer = new EarlyStoppingGraphTrainer(esConf, model, trainIter);
		
		EarlyStoppingResult<ComputationGraph> result = trainer.fit();
		
		logger.info("Termination reason: " + result.getTerminationReason());
		logger.info("Termination details: " + result.getTerminationDetails());
		logger.info("Total epochs: " + result.getTotalEpochs());
		logger.info("Best epoch number: " + result.getBestModelEpoch());
		logger.info("Score at best epoch: " + result.getBestModelScore());
		
		ComputationGraph bestModel = result.getBestModel();
		testIter.reset();
		
		long start = System.nanoTime();
        Evaluation evaluation = bestModel.evaluate(testIter);
        long diff = System.nanoTime() - start;
        logger.info("complete in: " + PrettyTime.prettyNano(diff));
        
        logger.info(evaluation.stats());
	}
	
	private static String getModelDir(){
		String dir = DataHelper.MODELS_DIR + dateFormat.format(System.currentTimeMillis()) + "/";
		return dir;
	}
	
	public void save(File file) throws IOException{
		save(file, false);
	}
	
	public void save(File file, boolean saveUpdater) throws IOException{
		ModelSerializer.writeModel(model, file, saveUpdater);
	}
	
	public Prediction predict(String sentence){
		
		INDArray result;
		if (nChannels == 1){
			INDArray features = singleChannelIter.loadSingleSentence(sentence);
			result = model.outputSingle(features);
		} else {
			INDArray[] features = multiChannelIter.loadSingleSentence(sentence);
			result = model.outputSingle(features);
		}
		
		return new Prediction(result.getDouble(0), result.getDouble(1));
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
		
		List<File> trainFiles = getSarcasmFiles(true);
		List<File> testFiles = getSarcasmFiles(false);
		
		logger.info("Reading word embeddings");
		List<WordVectors> embeddings = new ArrayList<WordVectors>();
//		embeddings.add(WordVectorSerializer.loadStaticModel(new File(DataHelper.GOOGLE_NEWS_WORD2VEC)));
		embeddings.add(WordVectorSerializer.loadStaticModel(new File(DataHelper.WORD2VEC_DIR + "all-preprocessed-300-test.emb")));
		logger.info("Reading word embedding - complete");


		
		int outputs = 2; 
		int batchSize = 32;
		int epochs = 5;
		int maxSentenceLength = 50;
		
		try {
			TextCNN cnn = new TextCNN(outputs, batchSize, epochs, embeddings, maxSentenceLength);
			cnn.startUIServer();
			long start = System.currentTimeMillis();
			cnn.train(trainFiles, testFiles);
			long end = System.currentTimeMillis();
			logger.info("Time taken: " + (end - start));
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
