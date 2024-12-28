import { Container, Typography, Box, Select, MenuItem, FormControl, InputLabel, Button } from "@mui/material";
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { useState } from 'react';
import { useDropzone } from 'react-dropzone';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#1976d2',
    },
  },
});

function App() {
  const [discriminator, setDiscriminator] = useState('');
  const [model, setModel] = useState('GPT 4o mini');
  const [file, setFile] = useState(null);
  const [fileContent, setFileContent] = useState('');
  
  const handleDiscriminatorChange = (event) => {
    setDiscriminator(event.target.value);
  };

  const handleModelChange = (event) => {
    setModel(event.target.value);
  };

  const handleFileRead = (file) => {
    const reader = new FileReader();

    reader.onload = (event) => {
      const fileContent = event.target.result;
      setFileContent(fileContent);
    };

    reader.readAsText(file);
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop: (acceptedFiles) => {
      const uploadedFile = acceptedFiles[0];
      setFile(uploadedFile);
      handleFileRead(uploadedFile);
    },
    accept: '.txt,.csv,.json',
  });

  const handleGenerateClick = async () => {
    if (fileContent) {
      try {
        // Send the GET request with file content, discriminator, and model
        const response = await fetch(`http://127.0.0.1:5000/generate?file_content=${encodeURIComponent(fileContent)}&discriminator=${discriminator}&model=${model}`);

        // Check if the response is successful
        if (response.ok) {
          const data = await response.json();
          console.log("Response from Flask backend:", data);

          // Create and download the data.txt file containing processed_data
          const processedData = data.processed_data;
          const blob = new Blob([processedData], { type: 'text/plain' });
          const link = document.createElement('a');
          link.href = URL.createObjectURL(blob);
          link.download = 'data.txt'; // The file name will be 'data.txt'
          link.click();
        } else {
          console.error("Error from Flask backend:", response.statusText);
        }
      } catch (error) {
        console.error("Error during fetch:", error);
      }
    } else {
      console.log("No file content to send.");
    }
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Container>
        <Box mt={10}>
          <Typography variant="h2">
            <span style={{ color: darkTheme.palette.primary.main }}>Meta</span>GAN-LLM
          </Typography>
        </Box>
        <Box mt={2}>
          <Typography variant="h6" color="textSecondary">
            Upload a file with data, and let our revolutionary GAN inspired LLM synthesize the rest.
          </Typography>
        </Box>
        
        {/* Discriminator and Model Dropdowns */}
        <Box mt={4} display="flex" gap={2}>
          <FormControl fullWidth sx={{ width: '15%' }}>
            <InputLabel id="discriminator-label">Discriminator</InputLabel>
            <Select
              labelId="discriminator-label"
              value={discriminator}
              onChange={handleDiscriminatorChange}
              label="Discriminator"
            >
              <MenuItem value="None">None</MenuItem>
              <MenuItem value="Statistical">Statistical</MenuItem>
              <MenuItem value="Adversarial">Adversarial</MenuItem>
              <MenuItem value="Explicit">Explicit</MenuItem>
            </Select>
          </FormControl>

          {/* Model Dropdown */}
          <FormControl fullWidth sx={{ width: '15%' }}>
            <InputLabel id="model-label">Model</InputLabel>
            <Select
              labelId="model-label"
              value={model}
              onChange={handleModelChange}
              label="Model"
            >
              <MenuItem value="GPT 4o mini">GPT 4o mini</MenuItem>
              <MenuItem value="GPT 4o">GPT 4o</MenuItem>
              <MenuItem value="Llama 3.1 8B">Llama 3.1 8B</MenuItem>
              <MenuItem value="Llama 3.1 70B">Llama 3.1 70B</MenuItem>
            </Select>
          </FormControl>
        </Box>

        {/* File Upload Area */}
        <Box mt={4}>
          <Typography variant="h6" color="textSecondary">
            Upload Your File
          </Typography>
          <Box
            {...getRootProps()}
            sx={{
              border: '2px dashed #1976d2',
              borderRadius: '8px',
              padding: '40px',
              textAlign: 'center',
              cursor: 'pointer',
              height: '400px', 
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <input {...getInputProps()} />
            <Typography variant="body1" color="textSecondary">
              {file ? `File: ${file.name}` : 'Drag and drop a file here, or click to select'}
            </Typography>
            <Button
              variant="contained"
              sx={{ marginTop: 2 }}
              onClick={() => document.querySelector('input[type="file"]').click()}
            >
              Choose File
            </Button>
          </Box>
        </Box>

        {/* Generate Button */}
        <Box mt={4} display="flex" justifyContent="flex-start">
          <Button
            variant="contained"
            color="primary"
            sx={{ padding: '10px 20px' }}
            onClick={handleGenerateClick} // Trigger the file generation and download
          >
            Generate
          </Button>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;
