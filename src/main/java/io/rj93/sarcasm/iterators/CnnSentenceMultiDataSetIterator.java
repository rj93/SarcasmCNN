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

@SuppressWarnings("serial")
public class CnnSentenceMultiDataSetIterator implements MultiDataSetIterator {
	
	private static final Logger logger = LogManager.getLogger(CnnSentenceMultiDataSetIterator.class);

    private static final String UNKNOWN_WORD_SENTINEL = "UNKNOWN_WORD_SENTINEL";

    private LabeledSentenceProvider sentenceProvider = null;
    private List<WordVectors> wordVectors;
    private TokenizerFactory tokenizerFactory;
    private UnknownWordHandling unknownWordHandling;
    private boolean useNormalizedWordVectors;
    private int minibatchSize;
    private int maxSentenceLength;
    private boolean sentencesAlongHeight;
    private MultiDataSetPreProcessor dataSetPreProcessor;
    private int channels;

    private List<Integer> wordVectorSizes;
    private int numClasses;
    private Map<String, Integer> labelClassMap;
    private INDArray unknown;
	
	public CnnSentenceMultiDataSetIterator(Builder builder) {
		this.sentenceProvider = builder.sentenceProvider;
        this.wordVectors = builder.wordVectors;
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
        
        wordVectorSizes = new ArrayList<Integer>(wordVectors.size());
        for (WordVectors wordVector : wordVectors){
        	wordVectorSizes.add(wordVector.getWordVector(wordVector.vocab().wordAtIndex(0)).length);
        }
        
        channels = wordVectors.size();
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
    	
        List<List<String>> tokens = new ArrayList<>();
        
        int[][] featuresShapes = new int[channels][4];
        for (int channel = 0; channel < channels; channel++){
        	tokens.add(tokenizeSentence(wordVectors.get(channel), sentence));
        	int[] featuresShape = new int[] {1, 1, 0, 0};
	        if (sentencesAlongHeight) {
	            featuresShape[2] = Math.min(maxSentenceLength, tokens.get(channel).size());
	            featuresShape[3] = wordVectorSizes.get(channel);
	        } else {
	            featuresShape[2] = wordVectorSizes.get(channel);
	            featuresShape[3] = Math.min(maxSentenceLength, tokens.size());
	        }
	        featuresShapes[channel] = featuresShape;
        }

        INDArray[] features = new INDArray[channels];
        for (int channel = 0; channel < channels; channel++){
        	features[channel] = Nd4j.create(featuresShapes[channel]);
	        int length = (sentencesAlongHeight ? featuresShapes[channel][2] : featuresShapes[channel][3]);
	        for (int i = 0; i < length; i++) {
	            INDArray vector = getVector(wordVectors.get(channel), tokens.get(channel).get(i));
	
	            INDArrayIndex[] indices = new INDArrayIndex[4];
	            indices[0] = NDArrayIndex.point(0);
	            indices[1] = NDArrayIndex.point(0);
	            if (sentencesAlongHeight) {
	                indices[2] = NDArrayIndex.point(i);
	                indices[3] = NDArrayIndex.all();
	            } else {
	                indices[2] = NDArrayIndex.all();
	                indices[3] = NDArrayIndex.point(i);
	            }
	
	            features[channel].put(indices, vector);
	        }
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
		
        List<Pair<List<List<String>>, String>> tokenizedSentences = new ArrayList<>(num); // list of pairs. pars = list of list of tokens, and label
        int[] maxLengths = new int[channels];
        for (int i = 0; i < channels; i++)
        	maxLengths[i] = -1;
        
        for (int i = 0; i < num && sentenceProvider.hasNext(); i++) {
            Pair<String, String> p = sentenceProvider.nextSentence();
            String sentence = p.getFirst();
            String label = p.getSecond();
            
            List<List<String>> pairTokens = new ArrayList<>(channels);
        	for (int j = 0; j < channels; j++){
	            List<String> tokens = tokenizeSentence(wordVectors.get(j), sentence);
	            pairTokens.add(tokens);
	
	            maxLengths[j] = Math.max(maxLengths[j], tokens.size());
        	}
        	tokenizedSentences.add(new Pair<>(pairTokens, label));
        }
        
        for (int i = 0; i < channels; i++){
	        if (maxSentenceLength > 0 && maxLengths[i] > maxSentenceLength) {
	            maxLengths[i] = maxSentenceLength;
	        }
        }
        
        int currMinibatchSize = tokenizedSentences.size();
        INDArray[] labels = {Nd4j.create(currMinibatchSize, numClasses)};
        for (int i = 0; i < tokenizedSentences.size(); i++) {
            String labelStr = tokenizedSentences.get(i).getSecond();
            if (!labelClassMap.containsKey(labelStr)) {
                throw new IllegalStateException("Got label \"" + labelStr + "\" that is not present in list of LabeledSentenceProvider labels");
            }

            int labelIdx = labelClassMap.get(labelStr);

            labels[0].putScalar(i, labelIdx, 1.0);
        }
        
        int[][] featuresShapes = new int[channels][4];
        for (int channel = 0; channel < channels; channel++){
	        int[] featuresShape = new int[4];
	        featuresShape[0] = currMinibatchSize;
	        featuresShape[1] = 1;
	        if (sentencesAlongHeight) {
	            featuresShape[2] = maxLengths[channel];
	            featuresShape[3] = wordVectorSizes.get(channel);
	        } else {
	            featuresShape[2] = wordVectorSizes.get(channel);
	            featuresShape[3] = maxLengths[channel];
	        }
	        featuresShapes[channel] = featuresShape;
        }
        
        INDArray[] features = new INDArray[channels];
        for (int channel = 0; channel < channels; channel++){
        	features[channel] = Nd4j.create(featuresShapes[channel]);
        	WordVectors wordVector = wordVectors.get(channel);
        	
        	for (int i = 0; i < currMinibatchSize; i++) {
        		List<String> currSentence = tokenizedSentences.get(i).getFirst().get(channel);
        		
        		for (int word = 0; word < currSentence.size() && word < maxSentenceLength; word++) {
        			INDArray vector = getVector(wordVector, currSentence.get(word));
        			
        			INDArrayIndex[] indices = new INDArrayIndex[4];
                    indices[0] = NDArrayIndex.point(i);
                    indices[1] = NDArrayIndex.point(0);
                    if (sentencesAlongHeight) {
                        indices[2] = NDArrayIndex.point(word);
                        indices[3] = NDArrayIndex.all();
                    } else {
                        indices[2] = NDArrayIndex.all();
                        indices[3] = NDArrayIndex.point(word);
                    }
                    
                    features[channel].put(indices, vector);
        		}
        	}
        }
        
        INDArray[] featuresMask = new INDArray[channels];
        for (int channel = 0; channel < channels; channel++){
        	featuresMask[channel] = Nd4j.create(currMinibatchSize, maxLengths[channel]);
        	
        	for (int i = 0; i < currMinibatchSize; i++) {
        		int sentenceLength = tokenizedSentences.get(i).getFirst().get(channel).size();
        		if (sentenceLength >= maxLengths[channel]) {
                    featuresMask[channel].getRow(i).assign(1.0);
        		} else {
        			featuresMask[channel].get(NDArrayIndex.point(i), NDArrayIndex.interval(0, sentenceLength)).assign(1.0);
        		}
        	}
        }
		
		
        MultiDataSet mds = new MultiDataSet(features, labels, featuresMask, null);
        if (dataSetPreProcessor != null)
        	dataSetPreProcessor.preProcess(mds);
        
		return mds;
	}
	
	private INDArray getVector(WordVectors wordVector, String word) {
        INDArray vector;
        if (unknownWordHandling == UnknownWordHandling.UseUnknownVector && word == UNKNOWN_WORD_SENTINEL) { //Yes, this *should* be using == for the sentinel String here
        	vector = unknown;
        } else {
            if (useNormalizedWordVectors) {
                vector = wordVector.getWordVectorMatrixNormalized(word);
            } else {
                vector = wordVector.getWordVectorMatrix(word);
            }
        }
        return vector;
    }
	
	private List<String> tokenizeSentence(WordVectors wordVector, String sentence) {
        Tokenizer t = tokenizerFactory.create(sentence);

        List<String> tokens = new ArrayList<>();
        while (t.hasMoreTokens()) {
            String token = t.nextToken();
            if (!wordVector.hasWord(token)) {
                switch (unknownWordHandling) {
                    case RemoveWord:
                    	System.out.println("unkown word: " + token);
                        continue;
                    case UseUnknownVector:
                        token = UNKNOWN_WORD_SENTINEL;
                }
            }
            tokens.add(token);
        }
        return tokens;
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
        private List<WordVectors> wordVectors = new ArrayList<WordVectors>();
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
        public Builder wordVectors(List<WordVectors> wordVectors) {
            this.wordVectors = wordVectors;
            return this;
        }
        
        public Builder addWordVector(WordVectors wordVector){
        	this.wordVectors.add(wordVector);
        	return this;
        }
        
        public Builder addWordVector(List<WordVectors> wordVectors){
        	this.wordVectors.addAll(wordVectors);
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

        public CnnSentenceMultiDataSetIterator build() {
            if (wordVectors == null || wordVectors.size() == 0) {
                throw new IllegalStateException(
                                "Cannot build CnnSentenceDataSetIterator without a WordVectors instance");
            }

            return new CnnSentenceMultiDataSetIterator(this);
        }

        
	}

}
