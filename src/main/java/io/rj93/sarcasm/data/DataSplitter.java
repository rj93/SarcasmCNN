package io.rj93.sarcasm.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.text.sentenceiterator.SentenceIterator;

public class DataSplitter {
	
	private float train;
	private float validation;
	private float test;
	private int seed;
	private Random rand;
	
	private List<String> trainSet = new ArrayList<String>();
	private List<String> validationSet = new ArrayList<String>();
	private List<String> testSet = new ArrayList<String>();
	
	public DataSplitter(){
		train = 0.6f;
		validation = 0.2f;
		test = 0.2f;
		seed = 123;
		rand = new Random(seed);
	}
	
	public DataSplitter(float train, float validation, float test, int seed){
		if (train + validation + test > 1)
			throw new IllegalArgumentException("The sum of train, validation, and float cannot be greater than 1");
		
		this.train = train;
		this.validation = validation;
		this.test = test;
		this.seed = seed;
		rand = new Random(seed);
	}
	
	public List<String> getTrainSet(){
		return trainSet;
	}
	
	public List<String> getValidationSet(){
		return validationSet;
	}
	
	public List<String> getTestSet(){
		return testSet;
	}
	
	public void split(List<String> data){
		for (String s : data){
			place(s);
		}
	}
	
	public void split(SentenceIterator iter){
		while(iter.hasNext()){
			place(iter.nextSentence());
		}
	}
	
	private void place(String s){
		float f = rand.nextFloat();
		if (f < train){
			trainSet.add(s);
		} else if (f < (train + validation)){
			validationSet.add(s);
		} else if (f < (train + validation + test)){
			testSet.add(s);
		} 
	}
	
}
