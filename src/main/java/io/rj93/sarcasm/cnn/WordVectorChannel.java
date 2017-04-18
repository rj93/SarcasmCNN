package io.rj93.sarcasm.cnn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
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
		return getFeatureVectors(Arrays.asList(sentence)).getFeatures();
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
        
        
        INDArrayIndex[] indices = new INDArrayIndex[4];
        indices[1] = NDArrayIndex.point(0);
        if (sentencesAlongHeight) {
            indices[3] = NDArrayIndex.all();
        } else {
            indices[2] = NDArrayIndex.all();
        }
        
        INDArray features = Nd4j.create(featuresShape);
        for (int i = 0; i < sentences.size(); i++) {
        	
            List<String> currSentence = tokenizedSentences.get(i);
            indices[0] = NDArrayIndex.point(i);
            
            for (int j = 0; j < currSentence.size() && j < maxSentenceLength; j++) {
                INDArray vector = getVector(currSentence.get(j));

                if (sentencesAlongHeight) {
                    indices[2] = NDArrayIndex.point(j);
                } else {
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
