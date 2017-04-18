package io.rj93.sarcasm.cnn;

import java.util.List;

import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Channel {
	
	public abstract int getSize();
	
	public abstract INDArray getFeatureVector(String sentence);
	
	public abstract MultiResult getFeatureVectors(List<String> sentences);
	
	public static Channel loadFromConfig(JSONObject config){
		String type = config.getString("type");
		
		if (type.equals(WordVectorChannel.TYPE)){
			return WordVectorChannel.loadFromConfig(config);
		}
		
		return null;
	}
}
