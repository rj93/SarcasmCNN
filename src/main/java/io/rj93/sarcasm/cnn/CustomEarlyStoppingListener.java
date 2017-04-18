package io.rj93.sarcasm.cnn;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.nn.graph.ComputationGraph;

import io.rj93.sarcasm.utils.PrettyTime;

public class CustomEarlyStoppingListener implements EarlyStoppingListener<ComputationGraph> {
	
	Logger logger = LogManager.getLogger(CustomEarlyStoppingListener.class);
	
	private long startTime;
	private long epochStartTime;

	@Override
	public void onStart(EarlyStoppingConfiguration<ComputationGraph> esConfig, ComputationGraph net) {
		startTime = System.nanoTime();
		epochStartTime = startTime;
	}

	@Override
	public void onEpoch(int epochNum, double score, EarlyStoppingConfiguration<ComputationGraph> esConfig, ComputationGraph net) {
		long diff = System.nanoTime() - epochStartTime;
		logger.info("Epoch {} completed in: {}, with score: {}", epochNum, PrettyTime.prettyNano(diff), score);
	}

	@Override
	public void onCompletion(EarlyStoppingResult<ComputationGraph> esResult) {
		long diff = System.nanoTime() - startTime;
		logger.info("Early stopping completed in: {}, termination reason: {}", PrettyTime.prettyNano(diff), esResult.getTerminationReason());
	}

}
