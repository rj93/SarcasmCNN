package io.rj93.sarcasm.data;

public class Sentence {
	
	private String data;
	private boolean sarcastic;
	
	public Sentence(String data, boolean sarcastic){
		this.data = data;
		this.sarcastic = sarcastic;
	}
	
	public String getData(){
		return data;
	}
	
	public boolean isSarcastic(){
		return sarcastic;
	}
}
