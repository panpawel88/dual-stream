# CMake utility functions for downloading files
# Shared between main CMakeLists.txt and tests/CMakeLists.txt

# Common function for downloading model files
function(download_model_file MODEL_URL MODEL_PATH MODEL_NAME TIMEOUT_SECONDS)
    if(NOT EXISTS "${MODEL_PATH}")
        message(STATUS "Downloading ${MODEL_NAME}...")
        # Use SHOW_PROGRESS for downloads with timeout > 30 seconds
        if(${TIMEOUT_SECONDS} GREATER 30)
            file(DOWNLOAD
                "${MODEL_URL}"
                "${MODEL_PATH}"
                TIMEOUT ${TIMEOUT_SECONDS}
                STATUS DOWNLOAD_STATUS
                LOG DOWNLOAD_LOG
                SHOW_PROGRESS
            )
        else()
            file(DOWNLOAD
                "${MODEL_URL}"
                "${MODEL_PATH}"
                TIMEOUT ${TIMEOUT_SECONDS}
                STATUS DOWNLOAD_STATUS
                LOG DOWNLOAD_LOG
            )
        endif()
        
        list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
        if(NOT DOWNLOAD_RESULT EQUAL 0)
            list(GET DOWNLOAD_STATUS 1 DOWNLOAD_ERROR)
            message(WARNING "Failed to download ${MODEL_NAME}: ${DOWNLOAD_ERROR}")
            message(STATUS "Download log: ${DOWNLOAD_LOG}")
            # Remove empty/failed file
            if(EXISTS "${MODEL_PATH}")
                file(REMOVE "${MODEL_PATH}")
            endif()
        else()
            # Validate that the downloaded file is not empty
            file(SIZE "${MODEL_PATH}" MODEL_SIZE)
            if(MODEL_SIZE EQUAL 0)
                message(WARNING "Downloaded ${MODEL_NAME} is empty (0 bytes). Removing invalid file.")
                file(REMOVE "${MODEL_PATH}")
            else()
                message(STATUS "Successfully downloaded ${MODEL_NAME} (${MODEL_SIZE} bytes)")
            endif()
        endif()
    else()
        # Check if existing file is empty and re-download if necessary
        file(SIZE "${MODEL_PATH}" EXISTING_SIZE)
        if(EXISTING_SIZE EQUAL 0)
            message(STATUS "Existing ${MODEL_NAME} is empty, re-downloading...")
            file(REMOVE "${MODEL_PATH}")
            download_model_file("${MODEL_URL}" "${MODEL_PATH}" "${MODEL_NAME}" ${TIMEOUT_SECONDS})
        else()
            message(STATUS "${MODEL_NAME} already exists (${EXISTING_SIZE} bytes), skipping download")
        endif()
    endif()
endfunction()