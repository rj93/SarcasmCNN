package io.rj93.sarcasm;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import io.rj93.sarcasm.data.DataHelper;
import io.rj93.sarcasm.iterators.MultiFileLineSentenceIterator;

public class Main {
	
	public static void main(String[] args) throws Exception{
		String fileNonSarcyPath = DataHelper.DATA_DIR + "preprocessed_data/2015/RC_2015-01-non-sarcy.json";
		String fileSarcyPath = DataHelper.DATA_DIR + "preprocessed_data/2015/RC_2015-01-sarcy.json";
		List<File> files = Arrays.asList(new File(fileNonSarcyPath), new File(fileSarcyPath));
		files = DataHelper.getFilesFromDir(DataHelper.PREPROCESSED_DATA_DIR, true);
//        trainWord2Vec(files, false);
        
//        Word2Vec vec = WordVectorSerializer.readWord2VecModel(DataHelper.GOOGLE_NEWS_WORD2VEC);
//        System.out.println(vec.getWordVectorMatrix("like"));
        
//        DataSplitter ds = new DataSplitter();
//        ds.split(iter);
//        System.out.println(ds.getTrainSet().size());
//        System.out.println(ds.getValidationSet().size());
//        System.out.println(ds.getTestSet().size());

//		DataHelper.seperateToMultipleFiles(new File(fileNonSarcyPath), new File(DataHelper.DATA_DIR + "train/neg/"));
//		DataHelper.seperateToMultipleFiles(new File(fileSarcyPath), new File(DataHelper.DATA_DIR + "train/pos/"));
		
		compareEmbeddings();
        
	}
	
	public static void trainWord2Vec(List<File> files, boolean save) throws IOException{
		SentenceIterator iter = new MultiFileLineSentenceIterator(files);
		
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());
		
		System.out.println("Building model");
		Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(300)
                .epochs(1)
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

        System.out.println(vec.getWordVectorMatrix("like"));
        if (save)
        	WordVectorSerializer.writeWord2VecModel(vec, DataHelper.WORD2VEC_DIR + "all-preprocessed-300-test.emb");
		
		
		System.out.println("Closest Words: ");
        Collection<String> lst = vec.wordsNearest("like", 10);
        System.out.println(lst);
	}
	
	public static void compareEmbeddings(){
		Word2Vec google = WordVectorSerializer.readWord2VecModel(DataHelper.GOOGLE_NEWS_WORD2VEC);
		Word2Vec mine = WordVectorSerializer.readWord2VecModel(DataHelper.WORD2VEC_DIR + "all-preprocessed-300.emb");
		
		String[] mostCommonWords = {"like", "get", "dont", "peopl", "im", "one", "would", "know", "make", "go"};
		
		for (String word : mostCommonWords){
			if(google.hasWord(word)){
				Collection<String> googleList = google.wordsNearest(word, 10);
				Collection<String> myList = mine.wordsNearest(word, 10);
				
				System.out.println("Closests word to: " + word);
				System.out.println("Google: " + googleList);
				System.out.println("Mine: " + myList);
				System.out.println();
			}
		}
	}
	
}
