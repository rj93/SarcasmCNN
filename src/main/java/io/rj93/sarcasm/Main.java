package io.rj93.sarcasm;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import io.rj93.sarcasm.data.DataHelper;

public class Main {
	
	public static void main(String[] args) throws IOException{
		String fileNonSarcyPath = DataHelper.DATA_DIR + "preprocessed_data/2015/RC_2015-01-non-sarcy.json";
		String fileSarcyPath = DataHelper.DATA_DIR + "preprocessed_data/2015/RC_2015-01-sarcy.json";
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
		
		System.out.print("Fitting model... ");
		long start = System.currentTimeMillis();
		vec.fit();
		long timeTaken = (System.currentTimeMillis() - start) / 1000;
		System.out.println("Finished in " + timeTaken + " seconds");
		
		System.out.println("Closest Words: ");
        Collection<String> lst = vec.wordsNearest("like", 10);
        System.out.println(lst);
        
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .setMaxIter(1000)
                .stopLyingIteration(250)
                .learningRate(500)
                .useAdaGrad(false)
                .theta(0.5)
                .setMomentum(0.5)
                .normalize(true)
                .build();
        vec.lookupTable().plotVocab(tsne, 3, new File("test.csv"));
        
	}
	
	
}
