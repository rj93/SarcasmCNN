package io.rj93.sarcasm.cnn;

public class Prediction {
	
	private double probabilityPositive;
	private double probabilityNegative;
	
	public Prediction(double probabilityPositive, double probabilityNegative){
		this.probabilityPositive = probabilityPositive;
		this.probabilityNegative = probabilityNegative;
	}
	
	public boolean isPositive(){
		return probabilityPositive > probabilityNegative;
	}
	
	public boolean isNegative(){
		return probabilityNegative > probabilityPositive;
	}
}
