package io.rj93.sarcasm.cnn.channels;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class SentimentChannel extends Channel {
	
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
		return getFeatureVectors(Arrays.asList(sentence)).getFeatures();
	}

	@Override
	public MultiResult getFeatureVectors(List<String> sentences) {
		
//		int[] featureShape = new int[]{sentences.size(), nParts};
		int[] featureShape = new int[4];
        featureShape[0] = sentences.size();
        featureShape[1] = 1;
        featureShape[2] = 1;
        featureShape[3] = nParts;
		INDArray features = Nd4j.create(featureShape);
		// System.out.println("features: " + features);
		for (int i = 0; i < sentences.size(); i++) {
			String sentence = sentences.get(i);
			List<String> tokens = tokenizeSentence(sentence);
			
			int tokensPerPart = (int) Math.ceil(tokens.size() / (double) nParts);
			String[][] parts = new String[nParts][tokensPerPart];
			
			for (int partIndex = 0; partIndex < nParts; partIndex++){
				for (int tokenIndex = 0; tokenIndex < tokensPerPart; tokenIndex++){
					int tokenListIndex = (partIndex * tokensPerPart) + tokenIndex;
					parts[partIndex][tokenIndex] = tokens.get(tokenListIndex);
				}
			}
			
			double[] partScores = new double[nParts];
			for (int partIndex = 0; partIndex < nParts; partIndex++){
				double partScore = 0;
				for (int tokenIndex = 0; tokenIndex < tokensPerPart; tokenIndex++){
					partScore += getSentimentScore(parts[partIndex][tokenIndex]);
				}
				partScores[partIndex] = partScore;
			}
			// System.out.println("partScores: " + Arrays.toString(partScores));
				
			INDArray vector = Nd4j.zeros(1,3);
			for (int j = 0; j < nParts; j++){
				vector.putScalar(0, j, partScores[j]);
			}
			// System.out.println("vector: " + vector);
			INDArrayIndex[] indices = new INDArrayIndex[4];
			indices[0] = NDArrayIndex.point(i);
			indices[1] = NDArrayIndex.point(0);
			indices[2] = NDArrayIndex.all();
			indices[3] = NDArrayIndex.all();
			features.put(indices, vector);
			
		}
		// System.out.println("features: " + features);
		return new MultiResult(features, null, nParts);
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

	@Override
	public JSONObject getConfig() {
		// TODO Auto-generated method stub
		return null;
	}

}
