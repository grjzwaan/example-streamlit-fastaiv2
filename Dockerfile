FROM continuumio/miniconda3

# Install
COPY env.yml .
RUN conda env create -n env -f env.yml

# Copy the app
COPY app.py .

# Activate
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

# Test
RUN streamlit

# Open the port
EXPOSE 80

# Fire it up!
CMD ["streamlit", "run", "app.py", "--server.port", "80"]