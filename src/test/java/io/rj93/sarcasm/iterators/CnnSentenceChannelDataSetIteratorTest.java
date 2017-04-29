package io.rj93.sarcasm.iterators;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.CnnSentenceDataSetIterator.UnknownWordHandling;
import org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import io.rj93.sarcasm.cnn.channels.Channel;
import io.rj93.sarcasm.cnn.channels.WordVectorChannel;
import io.rj93.sarcasm.iterators.CnnSentenceChannelDataSetIterator;
import io.rj93.sarcasm.utils.DataHelper;

public class CnnSentenceChannelDataSetIteratorTest {
private static final int seed = 100;
	
	private static int maxSentenceLength = 10;
	private int batchSize = 32;
	
	private static List<String> sentences = new ArrayList<>();
	private static List<String> labels = new ArrayList<>();
	private static Channel channel1;
	private static Channel channel2;
	private static WordVectors wordVector1;
	private static WordVectors wordVector2;
	
	@BeforeClass
	public static void setup(){
		sentences.addAll(readFile(DataHelper.PREPROCESSED_DATA_DIR + "2014/test/pos/RC_2014-01.json"));
		for (int i = 0; i < sentences.size(); i++){
			labels.add("positive");
		}
		sentences.addAll(readFile(DataHelper.PREPROCESSED_DATA_DIR + "2014/test/neg/RC_2014-01.json"));
		for (int i = labels.size(); i < sentences.size(); i++){
			labels.add("negative");
		}
		
		wordVector1 = WordVectorSerializer.loadStaticModel(new File(DataHelper.GLOVE_SMALL));
		wordVector1.setUNK("UNK");
		wordVector2 = WordVectorSerializer.loadStaticModel(new File(DataHelper.GLOVE_MEDIUM));
		wordVector2.setUNK("UNK");
		
		channel1 = new WordVectorChannel(DataHelper.GLOVE_SMALL, false, UnknownWordHandling.UseUnknownVector, maxSentenceLength);
		channel2 = new WordVectorChannel(DataHelper.GLOVE_MEDIUM, false, UnknownWordHandling.UseUnknownVector, maxSentenceLength);
	}
	
	@Test
	public void SingleChannelTest(){
		
		DataSetIterator dsi = getDataSetIterator(sentences, labels, maxSentenceLength, batchSize, wordVector1);
		MultiDataSetIterator mdsi = getMultiDataSetIterator(sentences, labels, batchSize, channel1);
		
		while(dsi.hasNext() && mdsi.hasNext()){
			DataSet ds = dsi.next();
			INDArray dsf = ds.getFeatures();
			INDArray dsl = ds.getLabels();
			
			MultiDataSet mds = mdsi.next();
			INDArray mdsf = mds.getFeatures(0);
			INDArray mdsl = mds.getLabels(0);
			
			assertEquals("Features", dsf, mdsf);
			assertEquals("Labels", dsl, mdsl);
		}
		
	}

	@Test
	public void MultiChannelTest(){
		
		DataSetIterator dsi1 = getDataSetIterator(sentences, labels, maxSentenceLength, batchSize, wordVector1);
		DataSetIterator dsi2 = getDataSetIterator(sentences, labels, maxSentenceLength, batchSize, wordVector2);
		MultiDataSetIterator mdsi = getMultiDataSetIterator(sentences, labels, batchSize, channel1, channel2);
		
		while(dsi1.hasNext() && dsi2.hasNext() && mdsi.hasNext()){
			DataSet ds1 = dsi1.next();
			INDArray dsf1 = ds1.getFeatures();
			INDArray dsl1 = ds1.getLabels();
			
			DataSet ds2 = dsi2.next();
			INDArray dsf2 = ds2.getFeatures();
			INDArray dsl2 = ds2.getLabels();
			
			MultiDataSet mds = mdsi.next();
			INDArray[] mdsf = mds.getFeatures();
			INDArray mdsl = mds.getLabels(0);
			
			assertEquals("Features1", dsf1, mdsf[0]);
			assertEquals("Features2", dsf2, mdsf[1]);
			assertEquals("Labels1", dsl1, mdsl);
			assertEquals("Labels2", dsl2, mdsl);
		}
	}
	
	@Test
	public void loadSingleSentenceTest(){
		CnnSentenceDataSetIterator dsi1 = (CnnSentenceDataSetIterator) getDataSetIterator(sentences, labels, maxSentenceLength, batchSize, wordVector1);
		CnnSentenceDataSetIterator dsi2 = (CnnSentenceDataSetIterator) getDataSetIterator(sentences, labels, maxSentenceLength, batchSize, wordVector2);
		CnnSentenceChannelDataSetIterator mdsi = (CnnSentenceChannelDataSetIterator) getMultiDataSetIterator(sentences, labels, batchSize, channel1, channel2);
		
		for (String s : sentences){
			
			INDArray dsf1 = dsi1.loadSingleSentence(s);
			INDArray dsf2 = dsi2.loadSingleSentence(s);
			INDArray[] mdsf = mdsi.loadSingleSentence(s);
			
			assertEquals("Features1", dsf1, mdsf[0]);
			assertEquals("Features2", dsf2, mdsf[1]);
		}
	}
	
	public static DataSetIterator getDataSetIterator(List<String> sentences, List<String> labels, int maxSentenceLength, int batchSize, WordVectors wordVector){
		
		LabeledSentenceProvider sentenceProvider = new CollectionLabeledSentenceProvider(sentences, labels, new Random(seed));
		
		DataSetIterator iter = new CnnSentenceDataSetIterator.Builder()
				.sentenceProvider(sentenceProvider)
                .wordVectors(wordVector)
                .maxSentenceLength(maxSentenceLength)
                .minibatchSize(batchSize)
                .useNormalizedWordVectors(false)
                .unknownWordHandling(UnknownWordHandling.UseUnknownVector)
                .build();
		
		return iter;
	}
	
	private static MultiDataSetIterator getMultiDataSetIterator(List<String> sentences, List<String> labels, int batchSize, Channel... channels){
		
		LabeledSentenceProvider sentenceProvider = new CollectionLabeledSentenceProvider(sentences, labels, new Random(seed));
		
		MultiDataSetIterator iter = new CnnSentenceChannelDataSetIterator.Builder()
				.sentenceProvider(sentenceProvider)
                .wordVectors(Arrays.asList(channels))
                .minibatchSize(batchSize)
                .build();
		
		return iter;
	}
	
	private static List<String> readFile(String file){
		File f = new File(file);
		List<String> s = null;
		try {
			s = FileUtils.readLines(f, "UTF-8");
		} catch (IOException e) {
			e.printStackTrace();
		}
		return s;
	}
}
