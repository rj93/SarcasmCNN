package io.rj93.sarcasm.iterators;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.LabelAwareConverter;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.documentiterator.LabelAwareDocumentIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.interoperability.DocumentIteratorConverter;
import org.deeplearning4j.text.sentenceiterator.interoperability.SentenceIteratorConverter;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import io.rj93.sarcasm.examples.CnnSentenceDataSetIterator;
import io.rj93.sarcasm.examples.CnnSentenceDataSetIterator.Builder;
import io.rj93.sarcasm.examples.CnnSentenceDataSetIterator.UnknownWordHandling;
import lombok.NonNull;

@SuppressWarnings("serial")
public class CnnSentenceMultiDataSetIterator implements MultiDataSetIterator {
	
	public enum UnknownWordHandling {
        RemoveWord, UseUnknownVector
    }

    private static final String UNKNOWN_WORD_SENTINEL = "UNKNOWN_WORD_SENTINEL";

    private LabeledSentenceProvider sentenceProvider = null;
    private List<WordVectors> wordVectors;
    private TokenizerFactory tokenizerFactory;
    private UnknownWordHandling unknownWordHandling;
    private boolean useNormalizedWordVectors;
    private int minibatchSize;
    private int maxSentenceLength;
    private boolean sentencesAlongHeight;
    private DataSetPreProcessor dataSetPreProcessor;

    private int wordVectorSize;
    private int numClasses;
    private Map<String, Integer> labelClassMap;
    private INDArray unknown;

    private int cursor = 0;
	
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

//        this.wordVectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length // TODO
	}
	
	@Override
	public boolean hasNext() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public MultiDataSet next() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public MultiDataSet next(int num) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
		// TODO Auto-generated method stub

	}

	@Override
	public boolean resetSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean asyncSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void reset() {
		// TODO Auto-generated method stub

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
        private DataSetPreProcessor dataSetPreProcessor;
        
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
        public Builder dataSetPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
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
