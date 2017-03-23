package io.rj93.sarcasm.data;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

public class DataHelper {
	
	public final static String DATA_DIR = getDataDir();
	public final static String REDDIT_DATA_DIR = DATA_DIR + "reddit_data";
	public final static String SORTED_DATA_DIR = DATA_DIR + "output_data";
	public final static String PREPROCESSED_DATA_DIR = DATA_DIR + "preprocessed_data";
	
	private static String getDataDir(){
		String dir = null;
		if (System.getProperty("os.name").contains("Windows")){
			dir = "E:/";
		} else {
			dir = "/Users/richardjones/Documents/";
		}
		dir += "Project/data/";
		return dir;
	}
	
	public static List<File> getFilesFromDir(File dir) throws FileNotFoundException {
		return getFilesFromDir(dir, false);
	}
	
	public static List<File> getFilesFromDir(File dir, boolean recursive) throws FileNotFoundException {
		
		if (!dir.exists() || !dir.isDirectory())
			throw new FileNotFoundException("Directory '" + dir.getAbsolutePath() + "' either does not exist, or is not a directory");
		
		List<File> files = new ArrayList<File>();
		for (File f : dir.listFiles()) {
			if (f.isFile())
				files.add(f);
			else if (recursive)
				files.addAll(getFilesFromDir(f, true));
		}
		
		return files;
	}

}
