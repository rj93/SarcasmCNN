package io.rj93.sarcasm.cnn;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class SentimentChannel implements Channel {
	
	private int nParts;
	private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
	
	public SentimentChannel(int nParts){
		this.nParts = nParts;
	}

	@Override
	public int getSize() {
		return nParts;
	}

	@Override
	public INDArray getFeatureVector(String sentence) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public MultiResult getFeatureVectors(List<String> sentences) {
		
//		List<List<String>> tokenizedSentences = new ArrayList<>();
//		for (String s : sentences){
//			tokenizedSentences.add(tokenizeSentence(s));
//		}
		
		
		int[] featureShape = new int[]{1, nParts};	
		INDArray features = Nd4j.create(sentences.size());
		for (int i = 0; i < sentences.size(); i++) {
			String sentence = sentences.get(i);
			List<String> tokens = tokenizeSentence(sentence);
			
			int tokensPerPart = (int) Math.ceil(tokens.size() / (double) nParts);
			String[][] parts = new String[nParts][tokensPerPart];
			
			for (int partIndex = 0; partIndex < nParts; partIndex++){
				for (int tokenIndex = 0; tokenIndex < tokens.size(); tokenIndex++){
					int tokenListIndex = (partIndex * nParts) + tokenIndex;
					parts[partIndex][tokenIndex] = tokens.get(tokenListIndex);
				}
			}
			
			double[] partScores = new double[nParts];
			for (int partIndex = 0; partIndex < nParts; partIndex++){
				double partScore = 0;
				for (int tokenIndex = 0; tokenIndex < tokens.size(); tokenIndex++){
					partScore += getSentimentScore(parts[partIndex][tokenIndex]);
				}
				partScores[partIndex] = partScore;
			}
			INDArray vector = Nd4j.create(partScores,featureShape);
//			features.putScalar(i, vector);
			
		}
		
		
		return null;
	}
	
	private double getSentimentScore(String word){
		double score = 0;
		if (word.equals("good")){
			score = 1;
		} else if (word.equals("bad")){
			score = -1;
		}
		return score;
	}
	
	private List<String> tokenizeSentence(String sentence){
		Tokenizer t = tokenizerFactory.create(sentence);
		
		List<String> tokens = new ArrayList<>();
        while (t.hasMoreTokens()) {
        	tokens.add(t.nextToken());
        }
		return tokens;
	}
	
	

}
