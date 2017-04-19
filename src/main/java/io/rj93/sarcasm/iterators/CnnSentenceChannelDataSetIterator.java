package io.rj93.sarcasm.iterators;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import io.rj93.sarcasm.cnn.channels.Channel;
import io.rj93.sarcasm.cnn.channels.MultiResult;

@SuppressWarnings("serial")
public class CnnSentenceChannelDataSetIterator implements MultiDataSetIterator {
	
	private static final Logger logger = LogManager.getLogger(CnnSentenceChannelDataSetIterator.class);

    private LabeledSentenceProvider sentenceProvider = null;
    private List<Channel> channels;
    private int minibatchSize;
    private MultiDataSetPreProcessor dataSetPreProcessor;
    private int nChannels;

    private List<Integer> channelSizes;
    private int numClasses;
    private Map<String, Integer> labelClassMap;
	
	public CnnSentenceChannelDataSetIterator(Builder builder) {
		this.sentenceProvider = builder.sentenceProvider;
        this.channels = builder.channels;
        this.minibatchSize = builder.minibatchSize;
        this.dataSetPreProcessor = builder.dataSetPreProcessor;


        this.numClasses = this.sentenceProvider.numLabelClasses();
        this.labelClassMap = new HashMap<>();
        int count = 0;
        //First: sort the labels to ensure the same label assignment order (say train vs. test)
        List<String> sortedLabels = new ArrayList<>(this.sentenceProvider.allLabels());
        Collections.sort(sortedLabels);
        
        for (String s : sortedLabels) {
            this.labelClassMap.put(s, count++);
        }
        
        channelSizes = new ArrayList<Integer>(channels.size());
        for (Channel channel : channels){
        	channelSizes.add(channel.getSize());
        }
        
        nChannels = channels.size();
	}
	
	@Override
	public boolean hasNext() {
		if (sentenceProvider == null) {
            throw new UnsupportedOperationException("Cannot do next/hasNext without a sentence provider");
        }
        return sentenceProvider.hasNext();
	}
	
	/**
     * Generally used post training time to load a single sentence for predictions
     */
    public INDArray[] loadSingleSentence(String sentence) {
    	
    	INDArray[] features = new INDArray[nChannels];
    	for (int channel = 0; channel < nChannels; channel++){
    		features[channel] = channels.get(channel).getFeatureVector(sentence);
    	}

        return features;
    }

	@Override
	public MultiDataSet next() {
		return next(minibatchSize);
	}

	@Override
	public MultiDataSet next(int num) {
		
		if (sentenceProvider == null) {
            throw new UnsupportedOperationException("Cannot do next/hasNext without a sentence provider");
        }
		
		List<Pair<String, String>> pairs = new ArrayList<>();
		for (int i = 0; i < num && sentenceProvider.hasNext(); i++){
			pairs.add(sentenceProvider.nextSentence());
		}
		
		List<String> sentences = new ArrayList<String>();
		INDArray[] labels = {Nd4j.create(pairs.size(), numClasses)};
		for (int i = 0; i < pairs.size(); i++){
			Pair<String, String> pair = pairs.get(i);
			sentences.add(pair.getFirst());
			int labelIdx = labelClassMap.get(pair.getSecond());
			labels[0].putScalar(i, labelIdx, 1.0);
		}
		
		
		INDArray[] features = new INDArray[nChannels];
		INDArray[] featuresMask = new INDArray[nChannels];
		for (int channel = 0; channel < nChannels; channel++){
			MultiResult result = channels.get(channel).getFeatureVectors(sentences);
			features[channel] = result.getFeatures();
			featuresMask[channel] = result.getFeaturesMask();
		}
		
		
		MultiDataSet mds = new MultiDataSet(features, labels, featuresMask, null);
        if (dataSetPreProcessor != null)
        	dataSetPreProcessor.preProcess(mds);
        
		return mds;
	}

	@Override
	public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
		this.dataSetPreProcessor = preProcessor;
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public boolean asyncSupported() {
		return true;
	}

	@Override
	public void reset() {
        sentenceProvider.reset();
	}
	
	public Map<String, Integer> getLabelsMap(){
		return labelClassMap;
	}
	
	public static class Builder {
		
		private LabeledSentenceProvider sentenceProvider = null;
        private List<Channel> channels = new ArrayList<Channel>();
        private int minibatchSize = 32;
        private MultiDataSetPreProcessor dataSetPreProcessor;
        
        /**
         * Specify how the (labelled) sentences / documents should be provided
         */
        public Builder sentenceProvider(LabeledSentenceProvider labeledSentenceProvider) {
            this.sentenceProvider = labeledSentenceProvider;
            return this;
        }

        /**
         * Provide the WordVectors instance that should be used for training
         */
        public Builder wordVectors(List<Channel> channels) {
            this.channels = channels;
            return this;
        }
        
        public Builder addWordVector(Channel channel){
        	this.channels.add(channel);
        	return this;
        }
        
        public Builder addWordVector(List<Channel> channels){
        	this.channels.addAll(channels);
        	return this;
        }

        /**
         * Minibatch size to use for the DataSetIterator
         */
        public Builder minibatchSize(int minibatchSize) {
            this.minibatchSize = minibatchSize;
            return this;
        }

        /**
         * Optional DataSetPreProcessor
         */
        public Builder dataSetPreProcessor(MultiDataSetPreProcessor dataSetPreProcessor) {
            this.dataSetPreProcessor = dataSetPreProcessor;
            return this;
        }

        public CnnSentenceChannelDataSetIterator build() {
            if (channels == null || channels.size() == 0) {
                throw new IllegalStateException(
                                "Cannot build CnnSentenceDataSetIterator without any channels");
            }

            return new CnnSentenceChannelDataSetIterator(this);
        }

        
	}

}
