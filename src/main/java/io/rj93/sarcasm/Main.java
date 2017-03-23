package io.rj93.sarcasm;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import io.rj93.sarcasm.conf.Configuration;

public class Main {
	
	public static void main(String[] args) throws IOException{
		String fileNonSarcyPath = Configuration.getDataDir() + "preprocessed_data/2015/RC_2015-01-non-sarcy.json";
		String fileSarcyPath = Configuration.getDataDir() + "preprocessed_data/2015/RC_2015-01-sarcy.json";
		List<File> files = Arrays.asList(new File(fileNonSarcyPath), new File(fileSarcyPath));
		
		
		SentenceIterator iter = new MultiFileLineSentenceIterator(files);
		
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());
		
		System.out.println("Building model");
		Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(100)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();
		
		System.out.println("Fitting model");
		vec.fit();
		
		System.out.println("Finished");


	}
	
	
}
