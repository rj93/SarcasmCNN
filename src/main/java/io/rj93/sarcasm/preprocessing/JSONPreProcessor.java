package io.rj93.sarcasm.preprocessing;

import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.filefilter.FileFilterUtils;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.json.JSONObject;

import io.rj93.sarcasm.iterators.MultiFileLineSentenceIterator;
import io.rj93.sarcasm.data.DataHelper;
import io.rj93.sarcasm.data.DataSplitter;

public class JSONPreProcessor extends TextPreProcessor {

	public String preProcess(String json) {
		JSONObject comment = new JSONObject(json);
		return super.preProcess(comment.getString("body"));
	}
	
	public static void main(String[] args){
		File inputDir = new File(DataHelper.SORTED_DATA_DIR);
		
		File[] years = inputDir.listFiles((FileFilter) FileFilterUtils.directoryFileFilter()); // only list directories
		for (File year : years){
			for (String Class : new String[]{"pos", "neg"}){
				try {

					List<File> files = DataHelper.getFilesFromDir(year.getAbsolutePath() + "/" + Class);
					
					List<String> processedStrings = new ArrayList<String>();
					MultiFileLineSentenceIterator iter = new MultiFileLineSentenceIterator(files);
					iter.setPreProcessor(new JSONPreProcessor());
					while (iter.hasNext()){
						processedStrings.add(iter.nextSentence());
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
				} catch (IOException e) {
					e.printStackTrace();
				}
				
			}
		}
	}

}
