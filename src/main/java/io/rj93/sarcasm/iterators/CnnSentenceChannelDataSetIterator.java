package io.rj93.sarcasm.iterators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator.UnknownWordHandling;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import io.rj93.sarcasm.cnn.Channel;
import io.rj93.sarcasm.cnn.MultiResult;

@SuppressWarnings("serial")
public class CnnSentenceChannelDataSetIterator implements MultiDataSetIterator {
	
	private static final Logger logger = LogManager.getLogger(CnnSentenceChannelDataSetIterator.class);

    private static final String UNKNOWN_WORD_SENTINEL = "UNKNOWN_WORD_SENTINEL";

    private LabeledSentenceProvider sentenceProvider = null;
    private List<Channel> channels;
    private TokenizerFactory tokenizerFactory;
    private UnknownWordHandling unknownWordHandling;
    private boolean useNormalizedWordVectors;
    private int minibatchSize;
    private int maxSentenceLength;
    private boolean sentencesAlongHeight;
    private MultiDataSetPreProcessor dataSetPreProcessor;
    private int nChannels;

    private List<Integer> channelSizes;
    private int numClasses;
    private Map<String, Integer> labelClassMap;
    private INDArray unknown;
	
	public CnnSentenceChannelDataSetIterator(Builder builder) {
		this.sentenceProvider = builder.sentenceProvider;
        this.channels = builder.channels;
        this.tokenizerFactory = builder.tokenizerFactory;
        this.unknownWordHandling = builder.unknownWordHandling;
        this.useNormalizedWordVectors = builder.useNormalizedWordVectors;
        this.minibatchSize = builder.minibatchSize;
        this.maxSentenceLength = builder.maxSentenceLength;
        this.sentencesAlongHeight = builder.sentencesAlongHeight;
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
	
	public static class Builder {
		
		private LabeledSentenceProvider sentenceProvider = null;
        private List<Channel> channels = new ArrayList<Channel>();
        private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        private UnknownWordHandling unknownWordHandling = UnknownWordHandling.RemoveWord;
        private boolean useNormalizedWordVectors = true;
        private int maxSentenceLength = -1;
        private int minibatchSize = 32;
        private boolean sentencesAlongHeight = true;
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
         * The {@link TokenizerFactory} that should be used. Defaults to {@link DefaultTokenizerFactory}
         */
        public Builder tokenizerFactory(TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        /**
         * Specify how unknown words (those that don't have a word vector in the provided WordVectors instance) should be
         * handled. Default: remove/ignore unknown words.
         */
        public Builder unknownWordHandling(UnknownWordHandling unknownWordHandling) {
            this.unknownWordHandling = unknownWordHandling;
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
         * Whether normalized word vectors should be used. Default: true
         */
        public Builder useNormalizedWordVectors(boolean useNormalizedWordVectors) {
            this.useNormalizedWordVectors = useNormalizedWordVectors;
            return this;
        }

        /**
         * Maximum sentence/document length. If sentences exceed this, they will be truncated to this length by
         * taking the first 'maxSentenceLength' known words.
         */
        public Builder maxSentenceLength(int maxSentenceLength) {
            this.maxSentenceLength = maxSentenceLength;
            return this;
        }

        /**
         * If true (default): output features data with shape [minibatchSize, 1, maxSentenceLength, wordVectorSize]<br>
         * If false: output features with shape [minibatchSize, 1, wordVectorSize, maxSentenceLength]
         */
        public Builder sentencesAlongHeight(boolean sentencesAlongHeight) {
            this.sentencesAlongHeight = sentencesAlongHeight;
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
