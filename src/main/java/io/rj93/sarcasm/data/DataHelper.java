package io.rj93.sarcasm.data;

import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.apache.commons.io.filefilter.FileFileFilter;
import org.apache.commons.io.filefilter.FileFilterUtils;

public class DataHelper {
	
	public final static String DATA_DIR = getDataDir();
	public final static String REDDIT_RAW_DIR = DATA_DIR + "reddit_raw/";
	public final static String REDDIT_FILTERED_DIR = DATA_DIR + "reddit_filtered/";
	public final static String PREPROCESSED_DATA_DIR = DATA_DIR + "preprocessed_data/";
	public final static String PREPROCESSED_DATA_STEMMED_DIR = DATA_DIR + "preprocessed_data_stemmed/";
	public final static String TRAIN_DATA_DIR = DATA_DIR + "train/";
	public final static String EVAL_DATA_DIR = PREPROCESSED_DATA_DIR + "eval/";
	public final static String TEST_DATA_DIR = PREPROCESSED_DATA_DIR + "test/";
	public final static String WORD2VEC_DIR = DATA_DIR + "word2vec/";
	public final static String MODELS_DIR = DATA_DIR + "models/";
	
	
	public final static String GOOGLE_NEWS_WORD2VEC = WORD2VEC_DIR + "google/GoogleNews-vectors-negative300.bin";
	
	private static String getDataDir(){
		String dir = null;
		if (System.getProperty("os.name").contains("Windows")){
			dir = "C:/Users/Richard/Documents/";
		} else {
			dir = "/Users/richardjones/Documents/";
		}
		dir += "Project/data/";
		return dir;
	}
	
	public static List<File> getFilesFromDir(File dir, boolean recursive, FilenameFilter filter) throws FileNotFoundException {
		
		if (!dir.exists() || !dir.isDirectory())
			throw new FileNotFoundException("Directory '" + dir.getAbsolutePath() + "' either does not exist, or is not a directory");
		
		List<File> files = new ArrayList<File>();
		for (File f : dir.listFiles()) {
			if (f.isFile() && acceptFilter(f, filter))
				files.add(f);
			else if (recursive && !f.isFile())
				files.addAll(getFilesFromDir(f, true, filter));
		}
		
		return files;
	}
	
	public static List<File> getFilesFromDir(File dir, FileFilter filter, boolean recursive) throws FileNotFoundException {
		if (!dir.exists() || !dir.isDirectory())
			throw new FileNotFoundException("Directory '" + dir.getAbsolutePath() + "' either does not exist, or is not a directory");
		
		List<File> files = new ArrayList<File>();
		for (File f : dir.listFiles()) {
			if (f.isFile() && acceptFilter(f, filter))
				files.add(f);
			else if (recursive && !f.isFile())
				files.addAll(getFilesFromDir(f, filter, true));
		}
		
		return files;
	}
	
	private static boolean acceptFilter(File f, FilenameFilter filter){
		if (filter != null)
			return filter.accept(f.getParentFile(), f.getName());
		return true;
	}
	
	private static boolean acceptFilter(File f, FileFilter filter){
		if (filter != null)
			return filter.accept(f);
		return true;
	}
	
	public static void seperateToMultipleFiles(File inFile, File outDir) throws IOException{
		outDir.mkdirs();
		List<String> s = Files.readAllLines(Paths.get(inFile.getAbsolutePath()));
		int count = 0;
		for (int i = 0; i < s.size(); i++){
			String line = s.get(i);
			if (line.length() > 3){
				PrintWriter writer = new PrintWriter(outDir.getAbsolutePath() + "/" + i + ".json", "UTF-8");
				writer.println(line);
				writer.close();
			} else {
				count++;
			}
		}
		System.out.println("Skipped " + count + " files");
	}

}
