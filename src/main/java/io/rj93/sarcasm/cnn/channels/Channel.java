package io.rj93.sarcasm.cnn.channels;

import java.util.List;

import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Channel {
	
	/**
	 * @return the size of the channel
	 */
	public abstract int getSize();
	
	/**
	 * Gets the features for a single text
	 * @param sentence the text
	 * @return the features of the text
	 */
	public abstract INDArray getFeatureVector(String sentence);
	
	/**
	 * Gets the feature vectors and feature mask for a list of sentences
	 * @param sentences list of texts
	 * @return MutliResult contianing feature vectors and feature mask
	 */
	public abstract MultiResult getFeatureVectors(List<String> sentences);
	
	/**
	 * Loads the channel from the string config
	 * @param config
	 * @return the loaded channel
	 */
	public static Channel loadFromConfig(String config){
		return loadFromConfig(new JSONObject(config));
	}
	
	/**
	 * Loads the channel from the JSON config
	 * @param config
	 * @return the loaded channel
	 */
	public static Channel loadFromConfig(JSONObject config){
		String type = config.getString("type");
		
		if (type.equals(WordVectorChannel.TYPE)){
			return WordVectorChannel.loadFromConfig(config);
		}
		
		return null;
	}
	
	/**
	 * @return the JSON config of the channel
	 */
	public abstract JSONObject getConfig();
}
