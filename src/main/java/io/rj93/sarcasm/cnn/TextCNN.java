package io.rj93.sarcasm.cnn;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.FileUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.deeplearning4j.api.storage.StatsStorage;
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
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator.UnknownWordHandling;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.format.datetime.standard.DateTimeContextHolder;

import io.rj93.sarcasm.data.DataHelper;
import io.rj93.sarcasm.filters.TestFileFilter;
import io.rj93.sarcasm.filters.TrainFileFilter;

public class TextCNN {
	
	private static final Logger logger = LogManager.getLogger(TextCNN.class);
	private static final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
			
	private int nChannels;
	private int nOutputs;
	private int batchSize;
	private int nEpochs;
	private int iterations = 1;
	private int seed;
	private WordVectors embedding;
	private int vectorSize;
	private int maxSentenceLength;
	private int cnnLayerFeatureMaps = 100;
	private ComputationGraphConfiguration conf;
	private ComputationGraph model;
	private UIServer uiServer = null;
	
	public TextCNN(int nChannels, int nOutputs, int batchSize, int nEpochs, WordVectors embedding, int maxSentenceLength){
		this(nChannels, nOutputs, batchSize, nEpochs, embedding, maxSentenceLength, 12345);
	}
	
	public TextCNN(int nChannels, int nOutputs, int batchSize, int nEpochs, WordVectors embedding, int maxSentenceLength, int seed){
		this.nChannels = nChannels;
		this.nOutputs = nOutputs;
		this.batchSize = batchSize;
		this.nEpochs = nEpochs;
		
		if (embedding.getUNK() == null)
			embedding.setUNK("UNK");
		this.embedding = embedding;
		
		this.vectorSize = 300;
		this.maxSentenceLength = maxSentenceLength;
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
	            .addInputs("input")
	            .addLayer("cnn3", new ConvolutionLayer.Builder()
	            	.name("cnn3")
	                .kernelSize(3,vectorSize)
	                .stride(1,vectorSize)
	                .nIn(nChannels)
	                .nOut(cnnLayerFeatureMaps)
	                .adamMeanDecay(0.999)
	                .adamVarDecay(0.9)
	                .build(), "input")
	            .addLayer("cnn4", new ConvolutionLayer.Builder()
	            	.name("cnn4")
	                .kernelSize(4,vectorSize)
	                .stride(1,vectorSize)
	                .nIn(nChannels)
	                .nOut(cnnLayerFeatureMaps)
	                .adamMeanDecay(0.999)
	                .adamVarDecay(0.9)
	                .build(), "input")
	            .addLayer("cnn5", new ConvolutionLayer.Builder()
	            	.name("cnn5")
	                .kernelSize(5,vectorSize)
	                .stride(1,vectorSize)
	                .nIn(nChannels)
	                .nOut(cnnLayerFeatureMaps)
	                .adamMeanDecay(0.999)
	                .adamVarDecay(0.9)
	                .build(), "input")
	            .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")
	            .addLayer("globalPool", new GlobalPoolingLayer.Builder()
	            	.name("globalPool")
	                .poolingType(PoolingType.MAX)
	                .build(), "merge")
	            .addLayer("out", new OutputLayer.Builder()
	            	.name("out")
	                .lossFunction(LossFunctions.LossFunction.MCXENT)
	                .activation(Activation.SOFTMAX)
	                .nIn(3*cnnLayerFeatureMaps)
	                .nOut(nOutputs)
	                .adamMeanDecay(0.999)
	                .adamVarDecay(0.9)
	                .build(), "globalPool")
	            .setOutputs("out")
	            .build();
		
		return conf;
	}
	
	private DataSetIterator getDataSetIterator(List<File> files) throws FileNotFoundException{

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
		
		
		DataSetIterator iter = new CnnSentenceDataSetIterator.Builder()
        		.sentenceProvider(sentenceProvider)
                .wordVectors(embedding)
                .minibatchSize(batchSize)
                .maxSentenceLength(maxSentenceLength)
                .useNormalizedWordVectors(false)
                .unknownWordHandling(UnknownWordHandling.UseUnknownVector)
                .build();
        return iter;
    }
	
	public void train(List<File> trainFiles, List<File> testFiles) throws IOException {
		
		DataSetIterator trainIter = getDataSetIterator(trainFiles);
		DataSetIterator testIter = getDataSetIterator(testFiles);
		
		logger.info("Training Model...");
		for (int i = 0; i < nEpochs; i++){
			
			
			logger.info("Starting epoch " + i + "... ");
			long start = System.nanoTime();
			model.fit(trainIter);
			long diff = System.nanoTime() - start;
			logger.info("Epoch " + i + " complete in " + diff / 1000000 + " ms. Starting evaluation...");
			
			start = System.nanoTime();
            Evaluation evaluation = model.evaluate(testIter);
            diff = System.nanoTime() - start;
            logger.info("Evaluation complete in: " + diff / (1000 * 1000) + " ms");
            
            logger.info(evaluation.stats());
            
            trainIter.reset();
            testIter.reset();
		}
		logger.info("Training Complete");
	}
	
	public void trainTest(List<File> trainFiles, List<File> testFiles) throws IOException {
		
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
        logger.info("complete in: " + diff / (1000 * 1000) + "ms");
        
        logger.info(evaluation.stats());
	}
	
	private static String getModelDir(){
		String dir = DataHelper.MODELS_DIR + dateFormat.format(System.currentTimeMillis()) + "/";
		return dir;
	}
	
	
	
	public void startUIServer(){
		uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage(); 
		uiServer.attach(statsStorage);
		model.setListeners(new StatsListener(statsStorage));
	}
	
	public void stopUIServer(){
		if  (uiServer != null)
			uiServer.stop();
	}
	
	public void save(File file) throws IOException{
		save(file, false);
	}
	
	public void save(File file, boolean saveUpdater) throws IOException{
		ModelSerializer.writeModel(model, file, saveUpdater);
	}
	
	public void test(List<File> testFiles) throws FileNotFoundException {
	}
	
	public static void main(String[] args) throws IOException {
		
		logger.info("Reading word embedding");
		WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(DataHelper.WORD2VEC_DIR + "all-preprocessed-300-test.emb"));
		logger.info("Reading word embedding - complete");

		File dir = new File(DataHelper.PREPROCESSED_DATA_DIR + "/2015-quick");
		List<File> trainFiles = DataHelper.getFilesFromDir(dir, new TrainFileFilter(), true);
		List<File> testFiles = DataHelper.getFilesFromDir(dir, new TestFileFilter(), true);
		
		int channels = 1;
		int outputs = 2;
		int batchSize = 32;
		int epochs = 5;
		int maxSentenceLength = 50;
		
		TextCNN cnn = new TextCNN(channels, outputs, batchSize, epochs, wordVectors, maxSentenceLength);
		cnn.startUIServer();
		long start = System.currentTimeMillis();
		cnn.train(trainFiles, testFiles);
		long end = System.currentTimeMillis();
		logger.info("Time taken: " + (end - start));
		
		System.exit(0);
		
	}
}
