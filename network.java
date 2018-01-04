
import java.util.*;
import cern.colt.matrix.impl.*;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.random.*;
import cern.jet.random.engine.DRand;
import cern.jet.math.*;

class network {
	
	//Fields:
	// layer == the set of neurons per layer in network.
	// networkScale == the number of layers in network.
	// weights == an arrayList of 2d matrices where: rows = #neurons in current layer & columns = #neurons in previous layer. 
	// biases == arrayList of vectors of biases for each layer.
	// alg == Algebra object for matrix calculations in methods.	
	int[] layer;
	int networkScale;
	ArrayList<DenseDoubleMatrix2D> weights;
	ArrayList<DenseDoubleMatrix1D> biases;
	ArrayList<DenseDoubleMatrix1D> layerActivations;
	ArrayList<DenseDoubleMatrix1D> zValues;
	ArrayList<DenseDoubleMatrix2D> gradientWeights;
	ArrayList<DenseDoubleMatrix1D> gradientBiases;
	Algebra alg = new Algebra();
	
	
	public network(int[] neurons) {
		// A network is instantiated by creating a set of random weights and biases based on the specified network size.
		
		
		layer = neurons;
		networkScale = neurons.length;
		weights = new ArrayList<>();
		biases = new ArrayList<>();
		
		//Creates a normal distribution. Mean = 0. SD = 1. Seed = System time. 
		DRand rE = new DRand((int)System.currentTimeMillis());
		Normal normalDis = new Normal(0, 1, rE);
		
		for(int i = 1; i < networkScale; i++) {
			
			//Create matrices for our weights and biases. weights[0] corresponds to the 2D matrix of weights of the second layer, and so forth. 
			//The same goes for the biases.
			DenseDoubleMatrix2D tempWeight = new DenseDoubleMatrix2D(layer[i - 1],layer[i]);
			DenseDoubleMatrix1D tempBias = new DenseDoubleMatrix1D(layer[i - 1]);	
			
			//Populate matrices with random numbers from our distribution.
			tempWeight.assign(normalDis);
			tempBias.assign(normalDis);
			
			//Add matrices to appropriate lists.
			weights.add(tempWeight);
			biases.add(tempBias);			
		}
	}
	
	public DenseDoubleMatrix1D sigmoidFunction(DenseDoubleMatrix1D reference) {
		
		//Network is composed of sigmoid neurons. Method for sigmoid function. sig(z) = 1/(1 + e^-z). z = wx - b;
		DenseDoubleMatrix1D input = (DenseDoubleMatrix1D) reference.copy();
		
		for(int i = 0; i < input.size(); i++) {			
			input.set(i, 1/(1 + Math.exp(-1*input.get(i))));			
		}		
		return input;		
	}
	
	public double scalarSigmoidDerivative(double in){
		
		//Scalar version of simoid derivative function.
		return Math.exp(-1*in)/Math.pow((1 + Math.exp(-1*in)), 2);
		
	}
	
	public DenseDoubleMatrix1D sigmoidDerivative(DenseDoubleMatrix1D reference) {
		
		//Derivative of sigmoid function used in back propagation algorithm. (d/dz)sig(z) = (e^-z)/((1+e^-z)^2) = sig(z)*(1 - sig(z))
		DenseDoubleMatrix1D input = (DenseDoubleMatrix1D) reference.copy();
		
		for(int i = 0; i < input.size(); i++) {			
			input.set(i, Math.exp(-1*input.get(i))/Math.pow((1 + Math.exp(-1*input.get(i))), 2));			
		}
		return input;		
	}
	
	public void feedForwardLoop(DenseDoubleMatrix1D a) {
		
		//This method updates the ArrayLists of activations and z-values. 
		zValues.clear();
		layerActivations.clear();
		layerActivations.add(a);
		
		//Iterates through each layer of the network.
		for(int i = 0; i < networkScale - 1; i++) {
			// a = sigmoidFunction(weights*[output of previous layer] + biases).
			DenseDoubleMatrix1D z = (DenseDoubleMatrix1D) alg.mult(weights.get(i), a);
			z.assign(biases.get(i),Functions.plus);
			
			zValues.add(z);
			
			a = sigmoidFunction(z);
			layerActivations.add(a);
		}

	}
	
	public void backPropogation(ArrayList<ArrayList<DenseDoubleMatrix1D>> batch) {
		
		//Iterative algorithm that calculates the gradient of the cost function. Running method updates Object fields gradientWeights and gradientBiases.
		
		ArrayList<DenseDoubleMatrix2D> bufferWeights = new ArrayList(weights);
		gradientWeights = new ArrayList(weights);
		gradientBiases = new ArrayList(biases);
		
		for(int i = 0; i < gradientWeights.size(); i++) gradientWeights.get(i).assign(0);
		for(int i = 0; i < gradientBiases.size(); i++) gradientBiases.get(i).assign(0);
		
		for(int batchIndex = 0; batchIndex < batch.size(); batchIndex++) {
			//Update activations and z values for current input in batch.
			feedForwardLoop(batch.get(0).get(batchIndex));
		
			DenseDoubleMatrix1D deltaFinal = new DenseDoubleMatrix1D(layer[networkScale - 1]);
			ArrayList<DenseDoubleMatrix1D> deltaLayer = new ArrayList<DenseDoubleMatrix1D>();
			//Calculate delta^L.
			for(int i = 0; i < deltaFinal.size(); i++) {
				double bufferDouble = (layerActivations.get(layerActivations.size() - 1).get(i) - batch.get(1).get(batchIndex).get(i))
						 * (scalarSigmoidDerivative(zValues.get(zValues.size() - 1).get(i)));
				deltaFinal.set(i, bufferDouble);
			}
			
			deltaLayer.add(deltaFinal);
			//Calculate delta for each layer. delta^l.
			for(int L = networkScale - 2; L > 0; L--) {
				
				DenseDoubleMatrix1D bufferVec = hadamadProduct(
				(DenseDoubleMatrix1D) alg.mult(alg.transpose(weights.get(L - 1)), deltaLayer.get(L+1)),
				sigmoidDerivative(zValues.get(L - 1)) );
				
				deltaLayer.add(bufferVec);
			}
			//Reverse layer so indexes match with other lists.
			Collections.reverse(deltaLayer);
			
			for(int L = 1; L < networkScale; L++) {
				//Sum 
				alg.multOuter(layerActivations.get(L - 1), deltaLayer.get(L - 1), bufferWeights.get(L - 1));
				gradientWeights.get(L - 1).assign(bufferWeights.get(L - 1), Functions.plus);
				gradientBiases.get(L - 1).assign(layerActivations.get(L - 1), Functions.plus);
				
			}
		}
		
		for(int i = 0; i < gradientWeights.size(); i++) {
			
			gradientWeights.get(i).assign(Functions.mult(1/batch.size()));
			gradientBiases.get(i).assign(Functions.mult(1/batch.size()));
			
		}
		
	}
	
	public void gradientDescent(double learningRate, int epochs, int batchSize, MNISTData data) {
		
		
		for(int j = 0; j < epochs; j++) {
			
			ArrayList<ArrayList<DenseDoubleMatrix1D>> batch = data.getBatch(batchSize);
			backPropogation(batch);
			
			for(int i = 0; i < gradientWeights.size(); i++) {
			
				gradientWeights.get(i).assign(Functions.mult(learningRate));
				gradientBiases.get(i).assign(Functions.mult(learningRate));
		
			}
		
			for(int i = 0; i < gradientWeights.size(); i++) {
			
				weights.get(i).assign(gradientWeights.get(i), Functions.plus);
				biases.get(i).assign(gradientBiases.get(i), Functions.plus);
			
			}
			
		}
		
	}
	
	public DenseDoubleMatrix1D hadamadProduct(DenseDoubleMatrix1D vec1, DenseDoubleMatrix1D vec2){
		
		//Method for Hadamad Product (Schur's product) since it is not included in colt. Only 1d case. 
		DenseDoubleMatrix1D result = (DenseDoubleMatrix1D) vec1.copy();
		
		for(int i = 0; i < result.size(); i++) {
			result.set(i, vec1.get(i) * vec2.get(i));
		}
		
		return result;
	}
	
	
}