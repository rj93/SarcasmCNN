package io.rj93.sarcasm.cnn.channels;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MultiResult {
	
	private INDArray features;
	private INDArray featuresMask;
	private int maxLength;
	
	public MultiResult(INDArray features, INDArray featuresMask, int maxLength){
		this.features = features;
		this.featuresMask = featuresMask;
		this.maxLength = maxLength;
	}
	
	public INDArray getFeatures(){
		return features;
	}	
	
	public INDArray getFeaturesMask(){
		return featuresMask;
	}
	
	public int getMaxLength(){
		return maxLength;
	}
	
}
