package io.rj93.sarcasm.cnn.channels;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class SentimentChannel extends Channel {
	
    private static final String pathToSWN = "src/main/resources/SentiWordNet_3.0.0_20130122.txt";
    private static Map<String, Double> dictionay = loadDictionary();
	
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
		
		int[] featureShape = new int[4];
        featureShape[0] = sentences.size();
        featureShape[1] = 1;
        featureShape[2] = nParts;
        featureShape[3] = 1;
		INDArray features = Nd4j.create(featureShape);

		for (int i = 0; i < sentences.size(); i++) {
			String sentence = sentences.get(i);
			List<String> tokens = tokenizeSentence(sentence);
			
			int tokensPerPart = (int) Math.ceil(tokens.size() / (double) nParts);
			String[][] parts = new String[nParts][tokensPerPart];
			
			for (int partIndex = 0; partIndex < nParts; partIndex++){
				for (int tokenIndex = 0; tokenIndex < tokensPerPart; tokenIndex++){
					int tokenListIndex = (partIndex * tokensPerPart) + tokenIndex;
					String token;
					if (tokenListIndex < tokens.size())
						token = tokens.get(tokenListIndex);
					else
						token = "";
					parts[partIndex][tokenIndex] = token;
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
				
			INDArray vector = Nd4j.zeros(1,3);
			for (int j = 0; j < nParts; j++){
				vector.putScalar(0, j, partScores[j]);
			}

			INDArrayIndex[] indices = new INDArrayIndex[4];
			indices[0] = NDArrayIndex.point(i);
			indices[1] = NDArrayIndex.point(0);
			indices[2] = NDArrayIndex.all();
			indices[3] = NDArrayIndex.all();
			features.put(indices, vector);
			
		}

		return new MultiResult(features, null, nParts);
	}
	
	private double getSentimentScore(String word){
		double score = 0;
	    if(dictionay.get(word+"#n") != null)
	    	score = dictionay.get(word+"#n") + score;
	    if(dictionay.get(word+"#a") != null)
	    	score = dictionay.get(word+"#a") + score;
	    if(dictionay.get(word+"#r") != null)
	    	score = dictionay.get(word+"#r") + score;
	    if(dictionay.get(word+"#v") != null)
	    	score = dictionay.get(word+"#v") + score;
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

	private static Map<String, Double> loadDictionary(){
		Map<String, Double> dict = new HashMap<String, Double>();

		// From String to list of doubles.
		HashMap<String, HashMap<Integer, Double>> tempDictionary = new HashMap<String, HashMap<Integer, Double>>();

		BufferedReader csv = null;
		try {
			csv = new BufferedReader(new FileReader(pathToSWN));
			int lineNumber = 0;

			String line;
			while ((line = csv.readLine()) != null) {
				lineNumber++;

				// If it's a comment, skip this line.
				if (!line.trim().startsWith("#")) {
					// We use tab separation
					String[] data = line.split("\t");
					String wordTypeMarker = data[0];

					// Example line:
					// POS ID PosS NegS SynsetTerm#sensenumber Desc
					// a 00009618 0.5 0.25 spartan#4 austere#3 ascetical#2
					// ascetic#2 practicing great self-denial;...etc

					// Is it a valid line? Otherwise, through exception.
					if (data.length != 6) {
						throw new IllegalArgumentException(
								"Incorrect tabulation format in file, line: "
										+ lineNumber);
					}

					// Calculate synset score as score = PosS - NegS
					Double synsetScore = Double.parseDouble(data[2])
							- Double.parseDouble(data[3]);

					// Get all Synset terms
					String[] synTermsSplit = data[4].split(" ");

					// Go through all terms of current synset.
					for (String synTermSplit : synTermsSplit) {
						// Get synterm and synterm rank
						String[] synTermAndRank = synTermSplit.split("#");
						String synTerm = synTermAndRank[0] + "#"
								+ wordTypeMarker;

						int synTermRank = Integer.parseInt(synTermAndRank[1]);
						// What we get here is a map of the type:
						// term -> {score of synset#1, score of synset#2...}

						// Add map to term if it doesn't have one
						if (!tempDictionary.containsKey(synTerm)) {
							tempDictionary.put(synTerm,
									new HashMap<Integer, Double>());
						}

						// Add synset link to synterm
						tempDictionary.get(synTerm).put(synTermRank,
								synsetScore);
					}
				}
			}

			// Go through all the terms.
			for (Map.Entry<String, HashMap<Integer, Double>> entry : tempDictionary
					.entrySet()) {
				String word = entry.getKey();
				Map<Integer, Double> synSetScoreMap = entry.getValue();

				// Calculate weighted average. Weigh the synsets according to
				// their rank.
				// Score= 1/2*first + 1/3*second + 1/4*third ..... etc.
				// Sum = 1/1 + 1/2 + 1/3 ...
				double score = 0.0;
				double sum = 0.0;
				for (Map.Entry<Integer, Double> setScore : synSetScoreMap
						.entrySet()) {
					score += setScore.getValue() / (double) setScore.getKey();
					sum += 1.0 / (double) setScore.getKey();
				}
				score /= sum;

				dict.put(word, score);
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (csv != null) {
				try { csv.close();} catch (IOException e) {}
			}
		}
		return dict;
	}

	
	@Override
	public JSONObject getConfig() {
		// TODO Auto-generated method stub
		return null;
	}

}
