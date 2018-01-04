import java.io.*;
import cern.colt.matrix.impl.*;
import java.util.ArrayList;
import java.util.Collections;

//Class dealing with MNIST testing and training data.
class MNISTData {
	
	//magicNum == 2049 for labels(solutions). magicNum == 2051 for test set.
	private int magicNumberImages;
	private int magicNumberLabels;
	private int sizeLabels;
	private int sizeImages;
	private int rowsImages;
	private int columnsImages;
	private ArrayList<DenseDoubleMatrix1D> images = new ArrayList<DenseDoubleMatrix1D>();
	private ArrayList<DenseDoubleMatrix1D> labels = new ArrayList<DenseDoubleMatrix1D>();
	
	//Constructor. Takes MNIST data pathname as parameter. 	
	public MNISTData(String imagesPath, String labelsPath) throws IOException {
		
		//Bring in files and create buffered stream.
		File fileImages = new File(imagesPath);
		InputStream inputImages = new FileInputStream(fileImages);
		BufferedInputStream bufferImages = new BufferedInputStream(inputImages);
		DataInputStream rawImages = new DataInputStream(bufferImages);
		File fileLabels = new File(labelsPath);
		InputStream inputLabels = new FileInputStream(fileLabels);
		BufferedInputStream bufferLabels = new BufferedInputStream(inputLabels);
		DataInputStream rawLabels = new DataInputStream(bufferLabels);
		//Set Object fields.
		magicNumberImages = rawImages.readInt();
		magicNumberLabels = rawLabels.readInt();
		
		sizeLabels = rawLabels.readInt();
		sizeImages = rawImages.readInt();
		
		rowsImages = rawImages.readInt();
		columnsImages = rawImages.readInt();
		
		//Construct our data for Images: An ArrayList of 1d vectors comprised of doubles from 0 to 255.  
		int vecSize = rowsImages*columnsImages;
		int outerIndex = 0;		
		while(outerIndex < sizeImages) {
			//System.out.println(outerIndex);
			double[] bufferArray = new double[vecSize];
			for(int i = 0; i < vecSize; i++) {
				bufferArray[i] = rawImages.readUnsignedByte();
				//System.out.println(holderArray[i]);
			}
			DenseDoubleMatrix1D bufferVec = new DenseDoubleMatrix1D(bufferArray);
			images.add(bufferVec);
			outerIndex++;
		}
		
		//Creates vectors corresponding to the labels [0,9].
		outerIndex = 0;
		while(outerIndex < sizeLabels) {
			
			int bufferInt = rawLabels.readUnsignedByte();
			
			switch (bufferInt) {
			case 0 : double[] holderArray0 = {1,0,0,0,0,0,0,0,0,0}; 
			DenseDoubleMatrix1D holderVec0 = new DenseDoubleMatrix1D(holderArray0);
			labels.add(holderVec0);
			break;
			case 1 : double[] holderArray1 = {0,1,0,0,0,0,0,0,0,0}; 
			DenseDoubleMatrix1D holderVec1 = new DenseDoubleMatrix1D(holderArray1);
			labels.add(holderVec1);
			break;
			case 2 : double[] holderArray2 = {0,0,1,0,0,0,0,0,0,0}; 
			DenseDoubleMatrix1D holderVec2 = new DenseDoubleMatrix1D(holderArray2);
			labels.add(holderVec2);
			break;
			case 3 : double[] holderArray3 = {0,0,0,1,0,0,0,0,0,0}; 
			DenseDoubleMatrix1D holderVec3 = new DenseDoubleMatrix1D(holderArray3);
			labels.add(holderVec3);
			break;
			case 4 : double[] holderArray4 = {0,0,0,0,1,0,0,0,0,0}; 
			DenseDoubleMatrix1D holderVec4 = new DenseDoubleMatrix1D(holderArray4);
			labels.add(holderVec4);
			break;
			case 5 : double[] holderArray5 = {0,0,0,0,0,1,0,0,0,0}; 
			DenseDoubleMatrix1D holderVec5 = new DenseDoubleMatrix1D(holderArray5);
			labels.add(holderVec5);
			break;
			case 6 : double[] holderArray6 = {0,0,0,0,0,0,1,0,0,0}; 
			DenseDoubleMatrix1D holderVec6 = new DenseDoubleMatrix1D(holderArray6);
			labels.add(holderVec6);
			break;
			case 7 : double[] holderArray7 = {0,0,0,0,0,0,0,1,0,0}; 
			DenseDoubleMatrix1D holderVec7 = new DenseDoubleMatrix1D(holderArray7);
			labels.add(holderVec7);
			break;
			case 8 : double[] holderArray8 = {0,0,0,0,0,0,0,0,1,0}; 
			DenseDoubleMatrix1D holderVec8 = new DenseDoubleMatrix1D(holderArray8);
			labels.add(holderVec8);
			break;
			case 9 : double[] holderArray9 = {0,0,0,0,0,0,0,0,0,1}; 
			DenseDoubleMatrix1D holderVec9 = new DenseDoubleMatrix1D(holderArray9);
			labels.add(holderVec9);
			break;
			}
			
			outerIndex++;
		}
		
		rawLabels.close();
		rawImages.close();
		
	} //End of constructor
	
	
	//Returns a subset of images and their corresponding labels as a batch of desired size.  
	public ArrayList<ArrayList<DenseDoubleMatrix1D>> getBatch(int batchSize) {
		
		ArrayList<Integer> shuffler = new ArrayList<Integer>();
		ArrayList<DenseDoubleMatrix1D> batchImages = new ArrayList<DenseDoubleMatrix1D>();
		ArrayList<DenseDoubleMatrix1D> batchLabels = new ArrayList<DenseDoubleMatrix1D>();
		ArrayList<ArrayList<DenseDoubleMatrix1D>> batch = new ArrayList<ArrayList<DenseDoubleMatrix1D>>();
		
		if(batchSize > sizeLabels || batchSize < 1) {
			return null;
		} 
		else 
			for(int i = 0; i < sizeLabels; i++) {
				shuffler.add(i);
			}
		
		Collections.shuffle(shuffler);
		
		for(int i : shuffler) {
			
			batchImages.add(images.get(i));
			batchLabels.add(labels.get(i));
			
		}
		
		batch.add(batchImages);
		batch.add(batchLabels);
		
		return batch;
	}
	
}