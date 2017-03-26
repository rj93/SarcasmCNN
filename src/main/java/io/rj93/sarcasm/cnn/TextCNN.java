package io.rj93.sarcasm.cnn;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
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
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(nOutputs)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
                .backprop(true)
                .pretrain(false)
                .build();
        return conf;
	}
	
	public void train(Word2Vec embedding, List<File> files) throws IOException{
		System.out.println("Training Model...");
		model.fit(new TextDataSetIterator(embedding, files, batchSize, 500));
		System.out.println("Training Complete");
	}
	
	public void test(){
		
	}
	
	public void predict(){
		
	}
	
	public static void main(String[] args) throws IOException {
		
		Word2Vec embedding = WordVectorSerializer.readWord2VecModel(DataHelper.WORD2VEC_DIR + "jan-2015-300.emb");
		TextCNN cnn = new TextCNN(1, 2, 64, 10, 10);
		cnn.train(embedding, DataHelper.getFilesFromDir(DataHelper.PREPROCESSED_DATA_DIR, true));
		
	}
}
