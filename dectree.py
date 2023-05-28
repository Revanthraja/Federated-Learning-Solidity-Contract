import solcx
import torch
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from web3 import Web3
import numpy as np
import struct
import joblib
# Solidity code
solidity_code = '''
pragma solidity ^0.8.0;

contract TrainingContract {
    // Define variables and data structures
    address[] public registeredNodes;
    string public sharedData;
    bytes[] public modelUpdates;
    string public aggregatedModel;

    // Node registration
    function registerNode() public {
        require(!isNodeRegistered(msg.sender), "Node is already registered");
        registeredNodes.push(msg.sender);
    }

    // Data sharing
    function shareData(string memory data) public {
        require(isNodeRegistered(msg.sender), "Node is not registered");
        sharedData = data;
    }

    // Model update submission
    function submitModelUpdate(bytes memory modelUpdate) public {
        require(isNodeRegistered(msg.sender), "Node is not registered");
        modelUpdates.push(modelUpdate);
    }

    // Model aggregation
    function aggregateModelUpdates() public {
        require(modelUpdates.length > 0, "No model updates available");

        bytes memory aggregated = new bytes(getTotalLength());
        uint256 currentIndex = 0;

        for (uint256 i = 0; i < modelUpdates.length; i++) {
            bytes memory update = modelUpdates[i];
            uint256 updateLength = update.length;

            for (uint256 j = 0; j < updateLength; j++) {
                aggregated[currentIndex] = update[j];
                currentIndex++;
            }
        }

        aggregatedModel = string(aggregated);

        delete modelUpdates;
    }

    // Get the length of registeredNodes array
    function registeredNodesLength() public view returns (uint256) {
        return registeredNodes.length;
    }

    // Helper function to check if a node is registered
    function isNodeRegistered(address node) private view returns (bool) {
        for (uint256 i = 0; i < registeredNodes.length; i++) {
            if (registeredNodes[i] == node) {
                return true;
            }
        }
        return false;
    }

    // Helper function to calculate the total length of all model updates
    function getTotalLength() private view returns (uint256) {
        uint256 totalLength = 0;
        for (uint256 i = 0; i < modelUpdates.length; i++) {
            totalLength += modelUpdates[i].length;
        }
        return totalLength;
    }
}


'''

ganache_url = "http://localhost:7545"
web3_instance = Web3(Web3.HTTPProvider(ganache_url))

# Step 1: Compile the Solidity code
compiled_contract = solcx.compile_source(solidity_code)
contract_abi = compiled_contract['<stdin>:TrainingContract']['abi']
contract_bytecode = compiled_contract['<stdin>:TrainingContract']['bin']

# Step 2: Deploy the Smart Contract
contract = web3_instance.eth.contract(abi=contract_abi, bytecode=contract_bytecode)
contract_constructor = contract.constructor()
deployed_contract = contract_constructor.build_transaction({
    'from': web3_instance.eth.accounts[0],
    'gas': 5000000,
    'gasPrice': web3_instance.to_wei('40', 'gwei')
})
transaction_hash = web3_instance.eth.send_transaction(deployed_contract)
transaction_receipt = web3_instance.eth.wait_for_transaction_receipt(transaction_hash)
contract_address = transaction_receipt['contractAddress']

# Step 3: Use the deployed contract for training
contract_instance = contract(address=contract_address)

# Step 4: Node Registration
node_address = web3_instance.eth.accounts[1]  # Address of the training node
contract_instance.functions.registerNode().transact({'from': node_address})

iris = load_iris()
data = iris.data

# Rest of the code...

# Step 6: Data Distribution
data_index = int(node_address, 16) % len(data)  # Convert the node address to an integer index
data_str = ','.join(str(value) for value in data[data_index])
contract_instance.functions.shareData(data_str).transact({'from': node_address})




decision_tree = RandomForestClassifier()

epochs = 10
for epoch in range(epochs):
    # Local model training using data[node_address]
    X = iris.data[epoch].reshape(1, -1)  # Use a single sample for each epoch
    y = np.array([iris.target[epoch]])  # Convert label to a 1D array with length 1
    decision_tree.fit(X, y)

    # Step 8: Model Update
    model_update = decision_tree.estimators_[0].tree_.value[0].flatten().tolist()
    model_update_bytes = struct.pack('f' * len(model_update), *model_update)
    contract_instance.functions.submitModelUpdate(model_update_bytes).transact({'from': node_address})



# Step 8: Model Aggregation
contract_instance.functions.aggregateModelUpdates().transact({'from': node_address})

# Step 9: Save the trained model
joblib.dump(decision_tree, 'trained_model.joblib')

# Step 10: Calculate and print the accuracy
X_test = data
y_test = iris.target
accuracy = decision_tree.score(X_test, y_test)
print("Accuracy:", accuracy)
