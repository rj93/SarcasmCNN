package io.rj93.sarcasm.preprocessing;

import org.json.JSONObject;

@SuppressWarnings("serial")
public class JSONPreProcessor extends TextPreProcessor {
	
	public JSONPreProcessor(){
		super();
	}

	public JSONPreProcessor(boolean removeStopWords, boolean stem) {
		super(removeStopWords, stem);
	}

	public String preProcess(String json) {
		try {
			JSONObject comment = new JSONObject(json);
			return super.preProcess(comment.getString("body"));
		} catch (Exception e){
			e.printStackTrace();
			System.err.println(json);
		}
		return "";
	}

}
