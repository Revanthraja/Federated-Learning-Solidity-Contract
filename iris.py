import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from solcx import compile_standard
from web3 import Web3

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the number of nodes in the network
num_nodes = 10

# Create a list to store the models for each node
nodes = []

# Train the models on each node
for node_id in range(num_nodes):
    model = RandomForestClassifier()
    start_idx = node_id * len(X_train) // num_nodes
    end_idx = (node_id + 1) * len(X_train) // num_nodes

    inputs = X_train[start_idx:end_idx]
    labels = y_train[start_idx:end_idx]

    model.fit(inputs, labels)
    nodes.append(model)

# Compile Solidity smart contracts using solcx
def compile_contract(contract_source_code):
    compiled_sol = compile_standard(
        {
            "language": "Solidity",
            "sources": {
                "contract.sol": {
                    "content": contract_source_code,
                }
            },
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ["abi", "evm.bytecode"]
                    }
                }
            }
        }
    )
    return compiled_sol

# Solidity smart contract source code
contract_source_code = """
pragma solidity ^0.8.0;

contract RandomForestClassifier {
    uint256 private constant NUM_FEATURES = 4;
    uint256 private constant NUM_CLASSES = 3;
    uint256 private constant NUM_NODES = 5;
    
    uint256[NUM_NODES][NUM_FEATURES] private nodeModels;

    function train(uint256[NUM_FEATURES] memory input, uint256 label) public {
        uint256 node_id = input[0] % NUM_NODES;
        
        for (uint256 i = 0; i < NUM_FEATURES; i++) {
            nodeModels[node_id][i] = input[i];
        }
    }

    function predict(uint256[NUM_FEATURES] memory input) public view returns (uint256) {
        uint256 node_id = input[0] % NUM_NODES;

        uint256 predictedClass = 0;
        uint256 maxVotes = 0;
        uint256[NUM_CLASSES] memory combinedPredictions;

        for (uint256 i = 0; i < NUM_NODES; i++) {
            uint256 nodeClass = input[0] % NUM_CLASSES;
            combinedPredictions[nodeClass] += 1;

            if (combinedPredictions[nodeClass] > maxVotes) {
                maxVotes = combinedPredictions[nodeClass];
                predictedClass = nodeClass;
            }
        }

        return predictedClass;
    }
}

"""

# Compile the Solidity smart contract
compiled_contract = compile_contract(contract_source_code)

# Get the compiled contract ABI and bytecode
contract_abi = compiled_contract["contracts"]["contract.sol"]["RandomForestClassifier"]["abi"]
contract_bytecode = compiled_contract["contracts"]["contract.sol"]["RandomForestClassifier"]["evm"]["bytecode"]["object"]

# Update with your local blockchain network URL
ganache_url = "http://localhost:7545"

# Connect to the local blockchain network
web3 = Web3(Web3.HTTPProvider(ganache_url))

# Set the default account (required for contract deployment)
account = web3.eth.accounts[0]  # Update with your desired account index
web3.eth.defaultAccount = account

# Deploy the compiled contract on the blockchain
contract = web3.eth.contract(abi=contract_abi, bytecode=contract_bytecode)
tx_hash = contract.constructor().transact({'from': account})
tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
contract_address = tx_receipt["contractAddress"]

print("Contract deployed at address:", contract_address)

# Configure communication and synchronization between nodes (e.g., using messaging or consensus mechanisms)
# In this example, we use a simple centralized approach where the main node distributes the model parameters to other nodes

# Define the main node as the leader
is_leader = True

# Distribute the data and training process among nodes
batch_size = len(X_train) // num_nodes  # Batch size per node

# Perform training on each node
for epoch in range(10):  # Number of training epochs
    # Iterate over the training data on each node
    for node_id in range(num_nodes):
        start_idx = node_id * batch_size
        end_idx = start_idx + batch_size

        inputs = X_train[start_idx:end_idx]
        labels = y_train[start_idx:end_idx]

        if is_leader:
            # Train the leader node's model
            nodes[node_id].fit(inputs, labels)
        else:
            # Train the other nodes' models
            # Call the contract's train function passing the node ID, inputs, and labels
            contract.functions.train(node_id, inputs[0], labels[0]).transact({'from': account})

    if is_leader:
        # Synchronize the model parameters among nodes
        global_weights = [node.get_params() for node in nodes]

        for node_id in range(num_nodes):
            # Set the updated model parameters on each node
            nodes[node_id].set_params(**global_weights[node_id])


    else:
        # Receive the updated model parameters from the leader node
        leader_params = contract.functions.getParams().call()
        nodes[0].set_params(leader_params)

# Evaluate the model on the test data
predictions = nodes[0].predict(X_test)
accuracy = sum(predictions == y_test) / len(y_test)
print("Test Accuracy:", accuracy)
