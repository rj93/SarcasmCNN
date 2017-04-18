package io.rj93.sarcasm.cnn;

import java.util.ArrayList;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator.UnknownWordHandling;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class WordVectorChannel implements Channel {
	
	Logger logger = LogManager.getLogger(WordVectorChannel.class);
	
	private static final String UNKNOWN_WORD_SENTINEL = "UNKNOWN_WORD_SENTINEL";
	
	private final WordVectors wordVector;
	private final boolean useNormalizedWordVectors;
	private final UnknownWordHandling unknownWordHandling;
	private final int size;
	private final boolean sentencesAlongHeight = true;
	private final int maxSentenceLength;
	private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
	private INDArray unknown;
	
	public WordVectorChannel(WordVectors wordVector, boolean useNormalizedWordVectors, UnknownWordHandling unknownWordHandling, int maxSentenceLength){
		this.wordVector = wordVector;
		size = wordVector.getWordVector(wordVector.vocab().wordAtIndex(0)).length;
		this.useNormalizedWordVectors = useNormalizedWordVectors;
		this.unknownWordHandling = unknownWordHandling;
		this.maxSentenceLength = maxSentenceLength;
	}

	@Override
	public int getSize() {
		return size;
	}
	
	@Override
	public INDArray getFeatureVector(String sentence) {

		List<String> tokens = tokenizeSentence(sentence);
		
		return getFeatureVector(tokens);
	}
	
	@Override
	public INDArray getFeatureVector(List<String> tokens) {
		
		int[] featuresShape = new int[] {1, 1, 0, 0};
        if (sentencesAlongHeight) {
            featuresShape[2] = Math.min(maxSentenceLength, tokens.size());
            featuresShape[3] = size;
        } else {
            featuresShape[2] = size;
            featuresShape[3] = Math.min(maxSentenceLength, tokens.size());
        }

        INDArray features = Nd4j.create(featuresShape);
        int length = (sentencesAlongHeight ? featuresShape[2] : featuresShape[3]);
        for (int i = 0; i < length; i++) {
            INDArray vector = getVector(tokens.get(i));

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

            features.put(indices, vector);
        }

        return features;
	}
	
	@Override
	public MultiResult getFeatureVectors(List<String> sentences) {
		
		List<List<String>> tokenizedSentences = new ArrayList<>(sentences.size());
        int maxLength = -1;
        int minLength = Integer.MAX_VALUE; //Track to we know if we can skip mask creation for "all same length" case
        for (int i = 0; i < sentences.size(); i++) {
            List<String> tokens = tokenizeSentence(sentences.get(i));

            maxLength = Math.max(maxLength, tokens.size());
            tokenizedSentences.add(tokens);
        }

        if (maxSentenceLength > 0 && maxLength > maxSentenceLength) {
            maxLength = maxSentenceLength;
        }

        int[] featuresShape = new int[4];
        featuresShape[0] = sentences.size();;
        featuresShape[1] = 1;
        if (sentencesAlongHeight) {
            featuresShape[2] = maxLength;
            featuresShape[3] = size;
        } else {
            featuresShape[2] = size;
            featuresShape[3] = maxLength;
        }

        INDArray features = Nd4j.create(featuresShape);
        for (int i = 0; i < sentences.size(); i++) {
            List<String> currSentence = tokenizedSentences.get(i);

            for (int j = 0; j < currSentence.size() && j < maxSentenceLength; j++) {
                INDArray vector = getVector(currSentence.get(j));

                INDArrayIndex[] indices = new INDArrayIndex[4];
                //TODO REUSE
                indices[0] = NDArrayIndex.point(i);
                indices[1] = NDArrayIndex.point(0);
                if (sentencesAlongHeight) {
                    indices[2] = NDArrayIndex.point(j);
                    indices[3] = NDArrayIndex.all();
                } else {
                    indices[2] = NDArrayIndex.all();
                    indices[3] = NDArrayIndex.point(j);
                }

                features.put(indices, vector);
            }
        }

        INDArray featuresMask = null;
        if (minLength != maxLength) {
            featuresMask = Nd4j.create(sentences.size(), maxLength);

            for (int i = 0; i < sentences.size(); i++) {
                int sentenceLength = tokenizedSentences.get(i).size();
                if (sentenceLength >= maxLength) {
                    featuresMask.getRow(i).assign(1.0);
                } else {
                    featuresMask.get(NDArrayIndex.point(i), NDArrayIndex.interval(0, sentenceLength)).assign(1.0);
                }
            }
        }
		
		return new MultiResult(features, featuresMask, maxLength);
		
	}
	
	private INDArray getVector(String word) {
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
	
	private List<String> tokenizeSentence(String sentence) {
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
	

}
