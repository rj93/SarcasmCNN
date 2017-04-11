package io.rj93.sarcasm.preprocessing;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.io.filefilter.FileFilterUtils;
import org.json.JSONArray;
import org.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.twelvemonkeys.io.FileUtil;

import io.rj93.sarcasm.data.DataHelper;
import io.rj93.sarcasm.examples.DataUtilities;

public class FilterFromRedditRaw {
	
	final private static Logger logger = LoggerFactory.getLogger(FilterFromRedditRaw.class);
	
	private static boolean decompressFile(String inPath, String outPath){
		boolean success = false;
		
		FileInputStream fin = null;
		BufferedInputStream in = null;
		FileOutputStream out = null;
		BZip2CompressorInputStream bzIn = null;
		try {
			fin = new FileInputStream(inPath);
			in = new BufferedInputStream(fin);
			out = new FileOutputStream(outPath);
			bzIn = new BZip2CompressorInputStream(in);
			final byte[] buffer = new byte[1024];
			int n = 0;
			while (-1 != (n = bzIn.read(buffer))) {
			    out.write(buffer, 0, n);
			}
			success = true;
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try { fin.close(); } catch (Exception e) { /* ignored */ }
			try { in.close(); } catch (Exception e) { /* ignored */ }
			try { out.close(); } catch (Exception e) { /* ignored */ }
			try { bzIn.close(); } catch (Exception e) { /* ignored */ }
		}
		if (success){
			logger.info("decompressFile: " + inPath + " successful");
		} else {
			logger.error("decompressFile: " + inPath + " failed!");
		}
		return success;
	}
	
	private static boolean deleteFile(String path){
		File f = new File(path);
		return f.delete();
	}
	
	private static void filter(String inFilePath, String outDirPath){
		File inFile = new File(inFilePath);

		String outPositivePath = outDirPath + "pos/";
		String outNegativePath = outDirPath + "neg/";
		File outPositiveFile = new File(outPositivePath);
		File outNegativeFile = new File(outNegativePath);
		outPositiveFile.mkdirs();
		outNegativeFile.mkdirs();
		
		List<Integer> sarcasticSizes = new ArrayList<Integer>();
		
		BufferedReader br = null;
		PrintWriter positiveWriter = null;
		PrintWriter negativeWriter = null;
		int posCount = 0;
		int negCount = 0;
		try {
			br = new BufferedReader(new FileReader(inFile));
			positiveWriter = new PrintWriter(outPositivePath + inFile.getName(), "UTF-8");
			negativeWriter = new PrintWriter(outNegativePath + inFile.getName(), "UTF-8");
			
			String line;
		    while ((line = br.readLine()) != null) {
		    	JSONObject comment = new JSONObject(line);
		    	String body = comment.getString("body");
		    	
		    	if (body.endsWith(" /s")){ // body is sarcastic
		    		sarcasticSizes.add(body.length());
		    		positiveWriter.println(line);
		    		posCount++;
		    	} else if (sarcasticSizes.size() > 0){ // search for similar sized non-sarcastic comments
		    		for (int i = 0; i < sarcasticSizes.size(); i++){
		    			int size = sarcasticSizes.get(i);
		    			if (Math.abs(body.length() - size) <= (0.1 * size)){ // compare sizes to be within 10% of each other
		    				negativeWriter.println(line);
		    				sarcasticSizes.remove(i);
		    				negCount++;
		    				break;
		    			}
		    		}
		    	}
		    }

			logger.info("filter: {} completed, pos = {}, neg = {}", inFilePath, posCount, negCount);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			try { br.close(); } catch (Exception e) { /* ignored */ }
			try { positiveWriter.close(); } catch (Exception e) { /* ignored */ }
			try { negativeWriter.close(); } catch (Exception e) { /* ignored */ }
		}
		
	}
	
	public static void main(String[] args) {
		File redditDataDir = new File(DataHelper.REDDIT_DATA_DIR);
		
		File[] years = redditDataDir.listFiles((FileFilter) FileFilterUtils.directoryFileFilter()); // only list directories
		for (File year : years){

			// list compressed files
			File[] compressedFiles = year.listFiles(new FilenameFilter() {
				@Override
				public boolean accept(File dir, String name) {
					return name.endsWith(".bz2");
				}
			});
			
			for (File f : compressedFiles){
				String compressedfilePath = f.getAbsolutePath();
				String decompressedFilePath = compressedfilePath.replace("bz2", "json");
				
				if (decompressFile(compressedfilePath, decompressedFilePath)){
					filter(decompressedFilePath, DataHelper.SORTED_DATA_DIR + year.getName() + "/");
					
					deleteFile(decompressedFilePath);
				}
			}
		}
	}

}
