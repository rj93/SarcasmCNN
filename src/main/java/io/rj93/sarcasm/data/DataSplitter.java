package io.rj93.sarcasm.data;

import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.filefilter.FileFilterUtils;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;

import io.rj93.sarcasm.iterators.MultiFileLineSentenceIterator;
import io.rj93.sarcasm.preprocessing.JSONPreProcessor;

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
		if (train + validation + test != 1)
			throw new IllegalArgumentException("The sum of train, validation, and test must equal 1");
		
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
	
	private static boolean writeToFile(String filePath, List<String> data){
		boolean success = false;
		
		File f = new File(filePath);
		f.getParentFile().mkdirs();
		
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(f);
			for (String s : data){
				writer.println(s);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} finally {
			try { writer.close(); } catch (Exception e) { /* ignored */ }
		}
		
		return success;
	}
	
	public static void main(String[] args){
		File inputDir = new File(DataHelper.SORTED_DATA_DIR);
		
		File[] years = inputDir.listFiles((FileFilter) FileFilterUtils.directoryFileFilter()); // only list directories
		for (File year : years){
			for (String Class : new String[]{"pos", "neg"}){
				try {

					List<File> files = DataHelper.getFilesFromDir(year.getAbsolutePath() + "/" + Class);
					for (File f : files){
						
						List<String> processedStrings = new ArrayList<String>();
						FileSentenceIterator iter = new FileSentenceIterator(f);
						iter.setPreProcessor(new JSONPreProcessor());
						while (iter.hasNext()){
							String s = iter.nextSentence();
//							System.out.println(s);
							processedStrings.add(s);
						}
						
						DataSplitter splitter = new DataSplitter(0.8f, 0f, 0.2f, 123);
						splitter.split(processedStrings);
						
						List<String> train = splitter.getTrainSet();
						List<String> val = splitter.getValidationSet();
						List<String> test = splitter.getTestSet();
						
						float totalSize = train.size() + val.size() + test.size();

						float trainPer = train.size() / totalSize;
						float valPer = val.size() / totalSize;
						float testPer = test.size() / totalSize;
						
						System.out.println(String.format("train = %f, val = %f, test = %f, dir = %s", trainPer, valPer, testPer, year.getName()+"/"+Class));
						
						String outDir = DataHelper.PREPROCESSED_DATA_DIR + year.getName() + "/";
						writeToFile(outDir + "train/" + Class + "/" + f.getName(), train);
						writeToFile(outDir + "test/" + Class + "/" + f.getName(), test);
					}
					
					
				} catch (IOException e) {
					e.printStackTrace();
				}
				
			}
		}
	}
	
}
