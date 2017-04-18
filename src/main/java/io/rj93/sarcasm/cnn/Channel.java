package io.rj93.sarcasm.cnn;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Channel {
	
	public int getSize();
	
	public INDArray getFeatureVector(String sentence);
	
	public INDArray getFeatureVector(List<String> sentences);
	
	public MultiResult getFeatureVectors(List<String> sentences);
	
}
