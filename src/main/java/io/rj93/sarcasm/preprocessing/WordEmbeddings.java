package io.rj93.sarcasm.preprocessing;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;

import io.rj93.sarcasm.iterators.MultiFileLineSentenceIterator;
import io.rj93.sarcasm.utils.DataHelper;
import io.rj93.sarcasm.utils.filters.TestFileFilter;
import io.rj93.sarcasm.utils.filters.TrainFileFilter;

public class WordEmbeddings {
	
	public static void main(String[] args) throws Exception{
		
//		File dir = new File(DataHelper.PREPROCESSED_DATA_STEMMED_DIR);
//		List<File> files = DataHelper.getFilesFromDir(dir, new TrainFileFilter(2), true);
//		files.addAll(DataHelper.getFilesFromDir(dir, new TestFileFilter(2), true));
//		trainWord2Vec(files, true);
		
//		compareEmbeddings();
		
//		buildTSNE();
	}
	
	public static void trainWord2Vec(List<File> files, boolean save) throws IOException{
		SentenceIterator iter = new MultiFileLineSentenceIterator(files);
		
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());
		
		System.out.println("Building model");
		Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(100)
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
        	WordVectorSerializer.writeWord2VecModel(vec, DataHelper.WORD2VEC_DIR + "all-preprocessed-stemmed-300.emb");
		
		System.out.println("Closest Words: ");
        Collection<String> lst = vec.wordsNearest("like", 10);
        System.out.println(lst);
        
	}
	
	public static void buildTSNE() throws IOException{
		
		System.out.print("Loading word2vec... ");
		WordVectors wordVector = WordVectorSerializer.readWord2VecModel(new File(DataHelper.WORD2VEC_DIR + "all-preprocessed-300.emb"));
		System.out.println("complete");
        VocabCache cache = wordVector.vocab();
        INDArray weights = wordVector.lookupTable().getWeights();
        
        List<String> cacheList = new ArrayList<String>();
		for(int i = 0; i < cache.numWords(); i++)   //seperate strings of words into their own list
        	cacheList.add(cache.wordAtIndex(i));
        
		System.out.print("Building TSNE... ");
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .setMaxIter(1)
                .theta(0.5)
                .normalize(false)
                .learningRate(500)
                .useAdaGrad(false)
                .numDimension(2)
//                .usePca(false)
                .build();
        System.out.println("complete");
        
        String outputFile = "src/main/resources/tsne-standard-coords.csv";
//        (new File(outputFile)).getParentFile().mkdirs();
        System.out.print("fitting TSNE... ");
        tsne.fit(weights);
        System.out.println("complete");

        System.out.println("writing to: " + outputFile);
        tsne.saveAsFile(cacheList, outputFile);
        System.out.println("finished");
	}
	
	public static void compareEmbeddings(){
		Word2Vec pretrained = WordVectorSerializer.readWord2VecModel(DataHelper.GLOVE);
		Word2Vec mine = WordVectorSerializer.readWord2VecModel(DataHelper.WORD2VEC_DIR + "all-preprocessed-300.emb");
		
		String[] mostCommonWords = {".", ",", "?", "!", "\"", "-", "*", "dont", ":", 
			"people", "im", "/", ";", "&", "yeah", "gt", "good", "...", "youre", 
			"time", "the", "make", "game", "didnt", "to", "a", "doesnt", "URL", "man", 
			"thing", "i", "back", "great", "hes", "ive", "work", "isnt", "and", "pretty", 
			"love", "2", "bad", "shit", "1", "you", "guy", "theyre", "fuck", "things", 
			"**", "3", "edit", "totally", "of", "made", "thought", "day", "guys", "lot", 
			"play", "makes", "years", "is", "women", "money", "%", "that", "year", 
			"ill", "feel", "post", "give", "fucking", "it", "life", "real", "id", "world", "point", 
			"put", "wrong", "$", "find", "reddit", "nice", "god", "in", "guess", "long", 
			"wow", "5", "person", "hard", "men", "big", "games", "white", "free", "read", "stop"};
		
		for (int i = 0; i < mostCommonWords.length; i++){
			String word = mostCommonWords[i];
			if(pretrained.hasWord(word)){
				Collection<String> pretrainedList = pretrained.wordsNearest(word, 10);
				Collection<String> myList = mine.wordsNearest(word, 10);
				
				System.out.println((i+1) + " - closests word to: " + word);
				System.out.println("GloVe: " + pretrainedList);
				System.out.println("Mine: " + myList);
				System.out.println();
			}
		}
	}
}
