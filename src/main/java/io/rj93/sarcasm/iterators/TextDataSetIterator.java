package io.rj93.sarcasm.iterators;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class TextDataSetIterator implements DataSetIterator {
	
	private final Word2Vec embedding;
	private final MultiFileLineSentenceIterator iterator;
	private final int batchSize;
    private final int size;
    private final TokenizerFactory tokenizerFactory;
    
	public TextDataSetIterator(Word2Vec embedding, List<File> files, int batchSize, int size) throws IOException{
		this.embedding = embedding;
		this.iterator = new MultiFileLineSentenceIterator(files);
		this.batchSize = batchSize;
		this.size = size;
		
		tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
	}
	
	public boolean hasNext() {
		return iterator.hasNext();
	}

	public DataSet next() {
		return next(batchSize);
	}

	public DataSet next(int num) {
		List<String> texts = new ArrayList<String>(num);
		boolean[] positive = new boolean[num];
		
		int count = 0;
		while(count < num && iterator.hasNext()){
			texts.add(iterator.nextSentence());
			positive[count] = iterator.isPositive();
			count++;
		}
		
		List<List<String>> allTokens = new ArrayList<List<String>>(texts.size());
        int maxLength = 0;
        for(String s : texts){
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            List<String> tokensFiltered = new ArrayList<String>();
            for(String t : tokens ){
                if(embedding.hasWord(t)) tokensFiltered.add(t);
            }
            allTokens.add(tokensFiltered);
            maxLength = Math.max(maxLength,tokensFiltered.size());
        }

        //If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
        if(maxLength > size) maxLength = size;

        //Create data for training
        //Here: we have reviews.size() examples of varying lengths
        INDArray features = Nd4j.create(texts.size(), embedding.getLayerSize(), maxLength);
        INDArray labels = Nd4j.create(texts.size(), 2, maxLength);    //Two labels: positive or negative
        //Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(texts.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(texts.size(), maxLength);

        int[] temp = new int[2];
        for( int i=0; i < texts.size(); i++ ){
            List<String> tokens = allTokens.get(i);
            temp[0] = i;
            //Get word vectors for each word in review, and put them in the training data
            for( int j=0; j<tokens.size() && j<maxLength; j++ ){
                String token = tokens.get(j);
                INDArray vector = embedding.getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);

                temp[1] = j;
                featuresMask.putScalar(temp, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
            }
            System.out.println("i: " + i + " num: " + num + " lenght: " + positive.length + " texts size: " + texts.size());
            System.out.println(Arrays.toString(positive));
            int idx = (positive[i] ? 0 : 1);
            int lastIdx = Math.min(tokens.size(),maxLength);
            labels.putScalar(new int[]{i,idx,lastIdx-1},1.0);   //Set label: [0,1] for negative, [1,0] for positive
            labelsMask.putScalar(new int[]{i,lastIdx-1},1.0);   //Specify that an output exists at the final time step for this example
        }

        return new DataSet(features,labels,featuresMask,labelsMask);
	}

	public int totalExamples() {
		// TODO Auto-generated method stub
		return 0;
	}

	public int inputColumns() {
		return embedding.getLayerSize();
	}

	public int totalOutcomes() {
		return 2;
	}

	public boolean resetSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	public boolean asyncSupported() {
		// TODO Auto-generated method stub
		return true;
	}

	public void reset() {
		// TODO Auto-generated method stub

	}

	public int batch() {
		return batchSize;
	}

	public int cursor() {
		// TODO Auto-generated method stub
		return 0;
	}

	public int numExamples() {
		// TODO Auto-generated method stub
		return 0;
	}

	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		// TODO Auto-generated method stub

	}

	public DataSetPreProcessor getPreProcessor() {
		// TODO Auto-generated method stub
		return null;
	}

	public List<String> getLabels() {
		// TODO Auto-generated method stub
		return Arrays.asList("positive","negative");
	}

}
