package io.rj93.sarcasm.cnn;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator.UnknownWordHandling;
import org.deeplearning4j.nn.graph.ComputationGraph;

import io.rj93.sarcasm.cnn.channels.Channel;
import io.rj93.sarcasm.cnn.channels.WordVectorChannel;
import io.rj93.sarcasm.utils.DataHelper;
import io.rj93.sarcasm.utils.PrettyTime;

public class TextCNNEvaluation {
	
	private static int outputs = 2; 
	private static int batchSize = 32;
	private static int epochs = 20;
	private static int maxSentenceLength = 100;
	
	private static Channel myChannel = new WordVectorChannel(DataHelper.WORD2VEC_DIR + "all-preprocessed-300-test.emb", true, UnknownWordHandling.UseUnknownVector, maxSentenceLength);
	private static Channel googleChannel = new WordVectorChannel(DataHelper.GOOGLE_NEWS_WORD2VEC, true, UnknownWordHandling.UseUnknownVector, maxSentenceLength);
	private static Channel gloveChannel = new WordVectorChannel(DataHelper.GLOVE_MEDIUM, true, UnknownWordHandling.UseUnknownVector, maxSentenceLength);
	
			
	public static void main(String[] args) throws IOException {
		List<File> trainFiles = DataHelper.getSarcasmFiles(true);
		List<File> testFiles = DataHelper.getSarcasmFiles(false);
		
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
				.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(2, TimeUnit.HOURS))
		        .evaluateEveryNEpochs(1)
				.build();
		
		for (int i = 0; i < channels.size(); i++){
			List<Channel> currChannels = channels.get(i);
			System.out.println(currChannels);
			TextCNN cnn = new TextCNN(outputs, batchSize, epochs, channels.get(i));
			long start = System.nanoTime();
			cnn.train(trainFiles, testFiles, esConf);
			long diff = System.nanoTime() - start;
			System.out.println("Total time taken: " + PrettyTime.prettyNano(diff));
		}
		
	}
	
}
