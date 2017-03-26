package io.rj93.sarcasm.iterators;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
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
    private final int truncateLength;
    private final TokenizerFactory tokenizerFactory;
    
	public TextDataSetIterator(Word2Vec embedding, List<File> files, int batchSize, int truncateLength) throws IOException{
		this.embedding = embedding;
		this.iterator = new MultiFileLineSentenceIterator(files);
		this.batchSize = batchSize;
		this.truncateLength = truncateLength;
		
		tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
	}
	
	public boolean hasNext() {
		return iterator.hasNext();
	}

	public DataSet next() {
		// TODO Auto-generated method stub
		String text = iterator.nextSentence();
		boolean positive = iterator.isPositive();
		
		List<String> tokens = tokenizerFactory.create(text).getTokens();
//		List<String> tokensFiltered = new ArrayList<String>();
//		for(String token : tokens ){
//            if(embedding.hasWord(token)) tokensFiltered.add(token);
//        }
		INDArray features = Nd4j.create(1, embedding.getLayerSize(), truncateLength);
        INDArray labels = Nd4j.create(1, 2, truncateLength);   
		
        for (int i = 0; i < tokens.size(); i++){
        	System.out.println("here");
        	String token = tokens.get(i);
        	INDArray vectorMatrix = embedding.getWordVectorMatrix(token);
        	features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(i)}, vectorMatrix);
        	
        	int idx = (positive ? 0 : 1);
        	int lastIdx = Math.min(tokens.size(),truncateLength);
        	labels.putScalar(new int[]{0,idx,lastIdx-1},1.0);
        }
        
		return new DataSet(features,labels);
	}

	public DataSet next(int num) {
		// TODO Auto-generated method stub
		return null;
	}

	public int totalExamples() {
		// TODO Auto-generated method stub
		return 0;
	}

	public int inputColumns() {
		return embedding.getLayerSize();
	}

	public int totalOutcomes() {
		// TODO Auto-generated method stub
		return 0;
	}

	public boolean resetSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	public boolean asyncSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	public void reset() {
		// TODO Auto-generated method stub

	}

	public int batch() {
		// TODO Auto-generated method stub
		return 0;
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
		return null;
	}

}
