package io.rj93.sarcasm.cnn;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.deeplearning4j.iterator.CnnSentenceDataSetIterator;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.iterator.provider.FileLabeledSentenceProvider;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import io.rj93.sarcasm.data.DataHelper;
import io.rj93.sarcasm.iterators.TextDataSetIterator;

public class TextCNN {
	
	private int nChannels;
	private int nOutputs;
	private int batchSize;
	private int nEpochs;
	private int iterations;
	private int seed = 12345;
	private MultiLayerConfiguration conf;
	private MultiLayerNetwork model;
	
	public TextCNN(int nChannels, int nOutputs, int batchSize, int nEpochs, int iterations){
		this(nChannels, nOutputs, batchSize, nEpochs, iterations, 12345);
	}
	
	public TextCNN(int nChannels, int nOutputs, int batchSize, int nEpochs, int iterations, int seed){
		this.nChannels = nChannels;
		this.nOutputs = nOutputs;
		this.batchSize = batchSize;
		this.nEpochs = nEpochs;
		this.iterations = iterations;
		this.seed = seed;
		this.conf = getConf();
		model = new MultiLayerNetwork(conf);
        model.init();
	}
	
	private MultiLayerConfiguration getConf(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations) // Training iterations as above
                .convolutionMode(ConvolutionMode.Same) 
                .regularization(true).l2(0.0005)
                .learningRate(.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                		.nIn(nChannels)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                		.nIn(nChannels)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                		.nIn(nChannels)
                        .nOut(nOutputs)
                        .activation(Activation.SOFTMAX)
                        .build())
//                .setInputType(InputType.convolutional(28,28,1))
                .backprop(true)
                .pretrain(false)
                .build();
        return conf;
	}
	
	public void train(Word2Vec embedding, List<File> files) throws IOException{
		
		List<File> positiveFiles = new ArrayList<File>();
		List<File> negativeFiles = new ArrayList<File>();
		for (File f : files){
			if (f.getName().contains("non-sarcy")){
				negativeFiles.add(f);
			} else {
				positiveFiles.add(f);
			}
		}
		
		Random r = new Random(100);
		Map<String,List<File>> filesMap = new HashMap<>();
        filesMap.put("Positive", positiveFiles);
        filesMap.put("Negative", negativeFiles);
		
		LabeledSentenceProvider sentenceProvider = new FileLabeledSentenceProvider(filesMap, r);
		
		DataSetIterator iter = new CnnSentenceDataSetIterator.Builder()
				.sentenceProvider(sentenceProvider)
				.wordVectors(embedding)
				.minibatchSize(batchSize)
				.maxSentenceLength(100)
				.useNormalizedWordVectors(false)
				.build();
		
		System.out.println("Training Model...");
		model.fit(iter);
		System.out.println("Training Complete");
	}
	
	public void test(){
		
	}
	
	public void predict(){
		
	}
	
	public static void main(String[] args) throws IOException {
		// CnnSentenceDataSetIterator
//		Word2Vec embedding = WordVectorSerializer.readWord2VecModel(DataHelper.WORD2VEC_DIR + "all-preprocessed-300.emb");
		System.out.println("Reading word embedding");
		Word2Vec embedding = WordVectorSerializer.readWord2VecModel(DataHelper.GOOGLE_NEWS_WORD2VEC);
		System.out.println("Complete");
		
		TextCNN cnn = new TextCNN(1, 2, 64, 10, 10);
		cnn.train(embedding, DataHelper.getFilesFromDir(DataHelper.PREPROCESSED_DATA_DIR, true));
		
	}
}
