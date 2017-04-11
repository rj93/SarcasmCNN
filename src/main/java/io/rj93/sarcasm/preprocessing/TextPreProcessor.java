package io.rj93.sarcasm.preprocessing;

import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

import opennlp.tools.stemmer.PorterStemmer;

@SuppressWarnings("serial")
public class TextPreProcessor implements SentencePreProcessor {
	
	final private static PorterStemmer stemmer = new PorterStemmer();
	private static Set<String> stopWords = getStopWords();
	
	private boolean removeStopWords;
	private boolean stem;
	
	public TextPreProcessor(){
		this(true, true);
	}
	
	public TextPreProcessor(boolean removeStopWords, boolean stem) {
		this.removeStopWords = removeStopWords;
		this.stem = stem;
	}
	
	@Override
	public String preProcess(String sentence) {
		sentence = sentence.toLowerCase();
		
		sentence = sentence.replace(" /s", "");
		sentence = sentence.replaceAll("'", "");
		
		sentence = sentence.replaceAll("/?u/[\\w-]+", "<USER>"); // usernames 
		sentence = sentence.replaceAll("\\b(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]", "<URL>");
		sentence = sentence.replaceAll("\\p{Punct}+", " "); // punctuation 
		sentence = sentence.trim().replaceAll("\\s{2,}", " "); // excess spaces
		
		StringBuilder sb = new StringBuilder();
		for (String word : sentence.split("\\s")){
			if (removeStopWords && stopWords.contains(word))
				continue;
			
			if (stem)
				sb.append(stemmer.stem(word));
			else 
				sb.append(word);
		}
		
		return sb.toString();
	}
	
	private static Set<String> getStopWords() {
		Set<String> stopWords = null;
		try {
			stopWords = new HashSet<String>(FileUtils.readLines(new File("data/stopWords.txt")));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return stopWords;
	}
	
	public static void main(String[] args){
		
		TextPreProcessor tpp = new TextPreProcessor();
		
		System.out.println(tpp.preProcess("/u/hello"));
		System.out.println(tpp.preProcess("u/hello"));
		System.out.println(tpp.preProcess("u/hello-world"));
		System.out.println(tpp.preProcess("u/hello-world-123"));
		System.out.println(tpp.preProcess("u/hello-world-123_456"));
		System.out.println(tpp.preProcess("u/hello-world-123_456 says hello"));
		System.out.println(tpp.preProcess("the grey fox jumps over the river"));
		System.out.println(tpp.preProcess("the grey fox jumps over...the river"));
	}

}
