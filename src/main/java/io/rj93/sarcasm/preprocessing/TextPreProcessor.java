package io.rj93.sarcasm.preprocessing;

import java.io.IOException;
import java.util.Arrays;

import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.util.AttributeFactory;
import org.apache.lucene.util.AttributeImpl;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

import opennlp.tools.stemmer.PorterStemmer;

public class TextPreProcessor implements SentencePreProcessor {
	
	final private static PorterStemmer stemmer = new PorterStemmer();
	final private static CharArraySet stopWords = EnglishAnalyzer.getDefaultStopSet();
	
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
			
		}
		
		TokenStream tokenStream = new StandardTokenizer(AttributeFactory.DEFAULT_ATTRIBUTE_FACTORY);
		try {
			tokenStream.reset();
//			System.out.println(tokenStream.);
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try { tokenStream.end(); } catch (Exception e) { /* ignored */ }
			try { tokenStream.close(); } catch (Exception e) { /* ignored */ }
		}
		
		
		return sentence;
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
