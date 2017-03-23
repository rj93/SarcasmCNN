package io.rj93.sarcasm.conf;

public class Configuration {
	
	public static String getDataDir(){
		String dir = null;
		if (System.getProperty("os.name").contains("Windows")){
			dir = "E:/";
		} else {
			dir = "/Users/richardjones/Documents/";
		}
		dir += "Project/data/";
		return dir;
	}
	
}
