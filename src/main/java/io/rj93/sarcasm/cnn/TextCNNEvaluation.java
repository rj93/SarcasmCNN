package io.rj93.sarcasm.cnn;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator.UnknownWordHandling;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.iq80.leveldb.util.FileUtils;

import io.rj93.sarcasm.cnn.channels.Channel;
import io.rj93.sarcasm.cnn.channels.WordVectorChannel;
import io.rj93.sarcasm.utils.DataHelper;
import io.rj93.sarcasm.utils.PrettyTime;

public class TextCNNEvaluation {
	
	private static int outputs = 2; 
	private static int batchSize = 32;
	private static int epochs = 20;
	private static int maxSentenceLength = 100;
	
	public static void main(String[] args) throws IOException {
//		buildModels();
//		buildAndTestStemmedModel();
//		testModels();
//		testRedditComp();
//		testSarcasmV2();
	}
	
	public static void buildModels() throws IOException {
		
		Channel myChannel = new WordVectorChannel(DataHelper.WORD2VEC_DIR + "all-preprocessed-300.emb", true, UnknownWordHandling.UseUnknownVector, maxSentenceLength);
		Channel googleChannel = new WordVectorChannel(DataHelper.GOOGLE_NEWS_WORD2VEC, true, UnknownWordHandling.UseUnknownVector, maxSentenceLength);
		Channel gloveChannel = new WordVectorChannel(DataHelper.GLOVE, true, UnknownWordHandling.UseUnknownVector, maxSentenceLength);
		
		List<File> trainFiles = DataHelper.getSarcasmFiles(true, false);
		List<File> testFiles = DataHelper.getSarcasmFiles(false, false);
		
		List<List<Channel>> channels = new ArrayList<>();
		channels.add(Arrays.asList(myChannel));
		channels.add(Arrays.asList(googleChannel));
		channels.add(Arrays.asList(gloveChannel));
		channels.add(Arrays.asList(myChannel, googleChannel));
		channels.add(Arrays.asList(myChannel, gloveChannel));
		channels.add(Arrays.asList(googleChannel, gloveChannel));
		channels.add(Arrays.asList(myChannel, googleChannel, gloveChannel));
		
		EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
				.epochTerminationConditions(new MaxEpochsTerminationCondition(30))
				.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(1, TimeUnit.DAYS))
		        .evaluateEveryNEpochs(1)
		        .saveLastModel(true)
				.build();
		
		for (int i = 0; i < channels.size(); i++){
			List<Channel> currChannels = channels.get(i);
			System.out.println(currChannels);
			TextCNN cnn = new TextCNN(outputs, batchSize, epochs, channels.get(i));
			long start = System.nanoTime();
			cnn.train(trainFiles, testFiles, esConf);
			long diff = System.nanoTime() - start;
			System.out.println("Total time taken: " + PrettyTime.prettyNano(diff));
			
			Evaluation eval = cnn.test(testFiles);
			System.out.println(eval.stats());
		}
		
	}
	
	public static void buildAndTestStemmedModel() throws IOException {
		Channel myChannelStemmed = new WordVectorChannel(DataHelper.WORD2VEC_DIR + "all-preprocessed-300.emb", true, UnknownWordHandling.UseUnknownVector, maxSentenceLength);
		List<File> trainFiles = DataHelper.getSarcasmFiles(true, false);
		List<File> testFiles = DataHelper.getSarcasmFiles(false, false);
		
		EarlyStoppingConfiguration<ComputationGraph> esConf = new EarlyStoppingConfiguration.Builder<ComputationGraph>()
				.epochTerminationConditions(new MaxEpochsTerminationCondition(30))
				.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(1, TimeUnit.DAYS))
		        .evaluateEveryNEpochs(1)
		        .saveLastModel(true)
				.build();
		
		TextCNN cnn = new TextCNN(outputs, batchSize, epochs, Arrays.asList(myChannelStemmed));
		long start = System.nanoTime();
		cnn.train(trainFiles, testFiles, esConf);
		long diff = System.nanoTime() - start;
		System.out.println("Total time taken: " + PrettyTime.prettyNano(diff));
		
		Evaluation eval = cnn.test(testFiles);
		System.out.println(eval.stats());
	}
	
	public static void testModels() throws IOException {
		List<File> testFiles = DataHelper.getSarcasmFiles(false, false);
		
		List<File> files = FileUtils.listFiles(new File(DataHelper.MODELS_DIR));
		for (File f : files){
			if (f.isDirectory()){
				System.out.println(f.getAbsolutePath());
				try {
					TextCNN cnn = TextCNN.loadFromDir(f.getAbsolutePath(), "bestGraph.bin");
					System.out.println(cnn.test(testFiles).stats());
				} catch (Exception e){
					e.printStackTrace();
				}
			}
		}
		
	}
	
	public static void testRedditComp() throws IOException {
		Channel gloveChannel = new WordVectorChannel(DataHelper.GLOVE, true, UnknownWordHandling.UseUnknownVector, maxSentenceLength);
		
		Map<String, String> trainMap = DataHelper.getRedditCompDataSet(true);
		Map<String, String> testMap = DataHelper.getRedditCompDataSet(false);
		
		TextCNN cnn = new TextCNN(outputs, batchSize, epochs, gloveChannel);
		long start = System.nanoTime();
		cnn.train(trainMap, testMap);
		long diff = System.nanoTime() - start;
		System.out.println("Total time taken: " + PrettyTime.prettyNano(diff));
	}
	
	public static void testSarcasmV2() throws IOException {
		Channel gloveChannel = new WordVectorChannel(DataHelper.GLOVE, true, UnknownWordHandling.UseUnknownVector, maxSentenceLength);
		
		Map<String, String> trainMap = DataHelper.getSarcsasmV2Dataset(true);
		Map<String, String> testMap = DataHelper.getSarcsasmV2Dataset(false);
		
		TextCNN cnn = new TextCNN(outputs, batchSize, epochs, gloveChannel);
		long start = System.nanoTime();
		cnn.train(trainMap, testMap);
		long diff = System.nanoTime() - start;
		System.out.println("Total time taken: " + PrettyTime.prettyNano(diff));
	}
	
	
	
}
