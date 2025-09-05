#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

int main() {
    try {
        // Test reading the test configuration JSON
        std::ifstream file("tests/test_config.json");
        if (!file.is_open()) {
            std::cout << "ERROR: Failed to open test_config.json" << std::endl;
            return 1;
        }
        
        nlohmann::json config;
        file >> config;
        
        std::cout << "✅ JSON parsing successful!" << std::endl;
        std::cout << "Configuration loaded with " << config["test_suites"].size() << " test suites:" << std::endl;
        
        for (const auto& suite : config["test_suites"]) {
            std::cout << "  - " << suite["name"].get<std::string>() << " (" 
                     << suite["tests"].size() << " tests)" << std::endl;
        }
        
        std::cout << "✅ JSON restoration successful!" << std::endl;
        return 0;
        
    } catch (const nlohmann::json::exception& e) {
        std::cout << "ERROR: JSON parsing failed: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}