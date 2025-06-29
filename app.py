import streamlit as st
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import tempfile
import pickle
import shutil
from datetime import datetime
import logging
from reid_backend import Config, APN_Dataset, APN_Model, TripletLoss, PersonReIDTrainer, PersonReIDTester, custom_collate

# Configure logging
logging.basicConfig(level=logging.INFO, filename='reid_app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set page config
st.set_page_config(
    page_title="Person Re-Identification System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .error-metric {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class PersonReIDApp:
    """Main application class"""

    def __init__(self):
        self.config = Config()
        self.model = None
        self.device = self.config.DEVICE

        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'embeddings_db' not in st.session_state:
            st.session_state.embeddings_db = None
        if 'training_history' not in st.session_state:
            st.session_state.training_history = None

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        transform = transforms.Compose([
            transforms.Resize(self.config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        return transform(image).unsqueeze(0).to(self.device)

    def get_embedding(self, image):
        """Get embedding for a single image"""
        if self.model is None:
            return None

        img_tensor = self.preprocess_image(image)
        self.model.eval()
        with torch.no_grad():
            embedding = self.model(img_tensor)
        return embedding.squeeze().cpu().numpy()

    def euclidean_distance(self, emb1, emb2):
        """Calculate euclidean distance between embeddings"""
        return np.sqrt(np.sum((emb1 - emb2) ** 2))

    def load_model(self, model_path):
        """Load trained model"""
        try:
            self.model = APN_Model(self.config.EMBEDDING_DIM).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            st.session_state.model_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            logging.error(f"Model loading error: {str(e)}")
            return False

    def create_embeddings_database(self, images_dict, save_path='embeddings_db.pkl'):
        """Create embeddings database from uploaded images"""
        embeddings_db = {}
        progress_bar = st.progress(0)

        for i, (name, image) in enumerate(images_dict.items()):
            try:
                embedding = self.get_embedding(image)
                if embedding is not None:
                    embeddings_db[name] = {
                        'embedding': embedding,
                        'image': image
                    }
            except Exception as e:
                st.warning(f"Skipping invalid image {name}: {str(e)}")
                logging.warning(f"Database creation - skipping image {name}: {str(e)}")
                continue
            progress_bar.progress((i + 1) / len(images_dict))

        with open(save_path, 'wb') as f:
            pickle.dump(embeddings_db, f)

        st.session_state.embeddings_db = embeddings_db
        return embeddings_db

    def load_embeddings_database(self, load_path='embeddings_db.pkl'):
        """Load embeddings database from disk"""
        try:
            if os.path.exists(load_path):
                with open(load_path, 'rb') as f:
                    st.session_state.embeddings_db = pickle.load(f)
                st.success(f"Loaded database with {len(st.session_state.embeddings_db)} images")
                return True
            else:
                st.error("No existing database found")
                return False
        except Exception as e:
            st.error(f"Error loading database: {str(e)}")
            logging.error(f"Database loading error: {str(e)}")
            return False

    def find_similar_persons(self, query_embedding, top_k=5):
        """Find similar persons in the database"""
        if st.session_state.embeddings_db is None:
            return []

        distances = []
        for name, data in st.session_state.embeddings_db.items():
            dist = self.euclidean_distance(query_embedding, data['embedding'])
            distances.append((name, dist, data['image']))

        distances.sort(key=lambda x: x[1])
        return distances[:top_k]

    def create_default_model(self):
        """Create and save a default model for initial testing"""
        try:
            model = APN_Model(self.config.EMBEDDING_DIM).to(self.device)
            model_path = 'default_model.pt'
            torch.save(model.state_dict(), model_path)
            
            if self.load_model(model_path):
                st.success("Default model created and loaded successfully!")
                st.info("You can now use the application with the default model, or upload your own trained model in Settings.")
                return True
            return False
        except Exception as e:
            st.error(f"Error creating default model: {str(e)}")
            logging.error(f"Default model creation error: {str(e)}")
            return False

    def check_and_load_model(self):
        """Check for existing models and load or create default"""
        # Try to load existing models in order of preference
        model_files = ['best_model.pt', 'trained_model.pt', 'default_model.pt']
        
        for model_file in model_files:
            if os.path.exists(model_file):
                if self.load_model(model_file):
                    st.info(f"Loaded existing model: {model_file}")
                    return True
        
        # If no model found, create default
        st.warning("No trained model found. Creating a default model for testing...")
        return self.create_default_model()

def main():
    """Main application function"""
    app = PersonReIDApp()

    st.markdown('<h1 class="main-header">👤 Person Re-Identification System</h1>', 
                unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔧 Model Training", "🔍 Person Search", "📊 Analytics", "⚙️ Settings"]
    )

    if page == "🏠 Home":
        show_home_page(app)
    elif page == "🔧 Model Training":
        show_training_page(app)
    elif page == "🔍 Person Search":
        show_search_page(app)
    elif page == "📊 Analytics":
        show_analytics_page(app)
    elif page == "⚙️ Settings":
        show_settings_page(app)

def show_home_page(app):
    """Home page"""
    st.markdown("## Welcome to the Person Re-Identification System")
    st.markdown("""
    This application uses deep learning to identify and match persons across different images.

    ### Key Features:
    - **Triplet Loss Training**: Advanced neural network training with triplet loss
    - **Real-time Search**: Search for similar persons in your database
    - **Comprehensive Testing**: 6 different test cases (TC01-TC06)
    - **Interactive Analytics**: Visualize embeddings and performance metrics
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.model_loaded:
            st.markdown('<div class="metric-card success-metric"><h4>✅ Model Status</h4><p>Loaded and Ready</p></div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card error-metric"><h4>❌ Model Status</h4><p>Not Loaded</p></div>', 
                       unsafe_allow_html=True)
            if st.button("🔧 Load/Create Default Model"):
                app.check_and_load_model()

    with col2:
        db_count = len(st.session_state.embeddings_db) if st.session_state.embeddings_db else 0
        st.markdown(f'<div class="metric-card metric-card"><h4>📊 Database</h4><p>{db_count} Images</p></div>', 
                   unsafe_allow_html=True)

    with col3:
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        st.markdown(f'<div class="metric-card metric-card"><h4>⚡ Device</h4><p>{device_info}</p></div>', 
                   unsafe_allow_html=True)

    st.markdown("## Quick Start Guide")
    st.markdown("""
    1. **Load Model**: Go to Settings and upload your trained model file
    2. **Create Database**: Upload reference images in the Person Search page
    3. **Search**: Upload a query image to find similar persons
    4. **Analyze**: View detailed analytics and performance metrics
    """)

def show_training_page(app):
    """Model training page"""
    st.markdown("## Model Training")

    st.markdown("### Training Configuration")
    col1, col2 = st.columns(2)

    with col1:
        epochs = st.slider("Epochs", 1, 50, app.config.EPOCHS)
        batch_size = st.slider("Batch Size", 8, 64, app.config.BATCH_SIZE)
        learning_rate = st.select_slider(
            "Learning Rate", 
            options=[0.001, 0.0005, 0.0001, 0.00005, 0.00001],
            value=app.config.LEARNING_RATE
        )

    with col2:
        margin = st.slider("Triplet Loss Margin", 0.1, 2.0, app.config.MARGIN, 0.1)
        embedding_dim = st.selectbox("Embedding Dimension", [64, 128, 256, 512], 
                                    index=1)

    st.markdown("### Dataset Upload")
    uploaded_csv = st.file_uploader("Upload training CSV file", type=['csv'])
    uploaded_images = st.file_uploader("Upload training images", type=['jpg', 'jpeg', 'png'], 
                                      accept_multiple_files=True)

    if uploaded_csv and uploaded_images:
        try:
            temp_dir = tempfile.mkdtemp()
            image_paths = {}
            for uploaded_file in uploaded_images:
                img_path = os.path.join(temp_dir, uploaded_file.name)
                with open(img_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                image_paths[uploaded_file.name] = img_path

            df = pd.read_csv(uploaded_csv)
            required_columns = ['Anchor', 'Positive', 'Negative']
            if not all(col in df.columns for col in required_columns):
                st.error("CSV must contain 'Anchor', 'Positive', 'Negative' columns")
                return

            for col in required_columns:
                for img_name in df[col]:
                    if img_name not in image_paths:
                        st.error(f"Image {img_name} not found in uploaded files")
                        return

            st.markdown("### Dataset Preview")
            st.dataframe(df.head())
            st.markdown(f"**Total samples**: {len(df)}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Anchor Images", len(df['Anchor'].unique()))
            with col2:
                st.metric("Positive Images", len(df['Positive'].unique()))
            with col3:
                st.metric("Negative Images", len(df['Negative'].unique()))

            if st.button("🚀 Start Training", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    app.config.EPOCHS = epochs
                    app.config.BATCH_SIZE = batch_size
                    app.config.LEARNING_RATE = learning_rate
                    app.config.MARGIN = margin
                    app.config.EMBEDDING_DIM = embedding_dim

                    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
                    trainset = APN_Dataset(train_df, temp_dir, app.config.INPUT_SIZE, augment=True)
                    validset = APN_Dataset(valid_df, temp_dir, app.config.INPUT_SIZE, augment=False)
                    trainloader = DataLoader(trainset, batch_size=app.config.BATCH_SIZE, shuffle=True,
                                           num_workers=app.config.NUM_WORKERS, collate_fn=custom_collate)
                    validloader = DataLoader(validset, batch_size=app.config.BATCH_SIZE, shuffle=False,
                                           num_workers=app.config.NUM_WORKERS, collate_fn=custom_collate)

                    model = APN_Model(app.config.EMBEDDING_DIM).to(app.config.DEVICE)
                    criterion = TripletLoss(margin=app.config.MARGIN)
                    optimizer = torch.optim.Adam(model.parameters(), lr=app.config.LEARNING_RATE)

                    trainer = PersonReIDTrainer(app.config)

                    def progress_callback(epoch, total_epochs, train_loss, valid_loss):
                        progress_bar.progress(epoch / total_epochs)
                        status_text.text(f"Epoch {epoch}/{total_epochs}: Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}")

                    train_losses, valid_losses = trainer.train(
                        model, trainloader, validloader, criterion, optimizer, app.config.EPOCHS, progress_callback)

                    st.session_state.training_history = {
                        'train_loss': train_losses,
                        'val_loss': valid_losses,
                        'epochs': list(range(1, len(train_losses) + 1))
                    }

                    model_path = 'trained_model.pt'
                    torch.save(model.state_dict(), model_path)
                    app.load_model(model_path)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=st.session_state.training_history['epochs'], 
                        y=st.session_state.training_history['train_loss'],
                        mode='lines+markers',
                        name='Training Loss',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=st.session_state.training_history['epochs'], 
                        y=st.session_state.training_history['val_loss'],
                        mode='lines+markers',
                        name='Validation Loss',
                        line=dict(color='red')
                    ))
                    fig.update_layout(
                        title='Training Progress',
                        xaxis_title='Epochs',
                        yaxis_title='Loss',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.success("Training completed!")

                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    st.info("Please check the dataset format and try again.")
                    logging.error(f"Training error: {str(e)}")

                finally:
                    if 'temp_dir' in locals():
                        shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            st.error(f"Error processing dataset: {str(e)}")
            logging.error(f"Dataset error: {str(e)}")

def show_search_page(app):
    """Person search page"""
    st.markdown("## Person Search")

    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load a model first in the Settings page!")
        return

    st.markdown("### Database Management")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_refs = st.file_uploader(
            "Upload reference images for the database", 
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="ref_upload"
        )

    with col2:
        if st.button("Load Existing Database"):
            app.load_embeddings_database()

    if uploaded_refs:
        if st.button("🗃️ Build Database"):
            with st.spinner("Building embeddings database..."):
                images_dict = {}
                for uploaded_file in uploaded_refs:
                    try:
                        image = Image.open(uploaded_file)
                        images_dict[uploaded_file.name] = np.array(image)
                    except Exception as e:
                        st.warning(f"Skipping invalid image {uploaded_file.name}: {str(e)}")
                        logging.warning(f"Database creation - skipping image {uploaded_file.name}: {str(e)}")
                        continue

                app.create_embeddings_database(images_dict)
                st.success(f"Database created with {len(images_dict)} images!")

    if st.session_state.embeddings_db:
        st.info(f"📊 Current database contains {len(st.session_state.embeddings_db)} images")

        if st.checkbox("Show database images"):
            cols = st.columns(5)
            for i, (name, data) in enumerate(list(st.session_state.embeddings_db.items())[:10]):
                with cols[i % 5]:
                    st.image(data['image'], caption=name, use_column_width=True)

    st.markdown("---")

    st.markdown("### Query Search")

    query_image = st.file_uploader(
        "Upload query image", 
        type=['jpg', 'jpeg', 'png'],
        key="query_upload"
    )

    if query_image is not None:
        col1, col2 = st.columns([1, 2])

        with col1:
            query_img = Image.open(query_image)
            st.image(query_img, caption="Query Image", use_column_width=True)

        with col2:
            top_k = st.slider("Number of results", 1, 10, 5)
            distance_threshold = st.slider("Distance threshold", 0.0, 2.0, 1.0, 0.1)

            if st.button("🔍 Search", type="primary"):
                if st.session_state.embeddings_db is None:
                    st.error("Please build the database first!")
                else:
                    with st.spinner("Searching..."):
                        query_embedding = app.get_embedding(np.array(query_img))

                        if query_embedding is not None:
                            results = app.find_similar_persons(query_embedding, top_k)

                            st.markdown("### Search Results")

                            if results:
                                cols = st.columns(min(3, len(results)))

                                for i, (name, distance, image) in enumerate(results):
                                    if distance <= distance_threshold:
                                        with cols[i % 3]:
                                            st.image(image, caption=f"{name}", use_column_width=True)

                                            if distance < 0.3:
                                                st.success(f"Distance: {distance:.3f} (Very Similar)")
                                            elif distance < 0.6:
                                                st.info(f"Distance: {distance:.3f} (Similar)")
                                            else:
                                                st.warning(f"Distance: {distance:.3f} (Possibly Similar)")

                                names = [r[0] for r in results if r[1] <= distance_threshold]
                                distances = [r[1] for r in results if r[1] <= distance_threshold]

                                if names:
                                    fig = go.Figure(data=[
                                        go.Bar(x=names, y=distances, 
                                              marker_color=['green' if d < 0.3 else 'orange' if d < 0.6 else 'red' 
                                                          for d in distances])
                                    ])
                                    fig.update_layout(
                                        title='Similarity Distances',
                                        xaxis_title='Images',
                                        yaxis_title='Distance',
                                        height=300
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No similar persons found within the distance threshold.")

def show_analytics_page(app):
    """Analytics page"""
    st.markdown("## Analytics Dashboard")

    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load a model first in the Settings page!")
        return

    if st.session_state.training_history:
        st.markdown("### Training Performance")
        col1, col2 = st.columns(2)

        with col1:
            history = st.session_state.training_history
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history['epochs'], 
                y=history['train_loss'],
                mode='lines+markers',
                name='Training Loss',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=history['epochs'], 
                y=history['val_loss'],
                mode='lines+markers',
                name='Validation Loss',
                line=dict(color='red')
            ))
            fig.update_layout(
                title='Training Curves',
                xaxis_title='Epochs',
                yaxis_title='Loss',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]
            st.metric("Final Training Loss", f"{final_train_loss:.4f}")
            st.metric("Final Validation Loss", f"{final_val_loss:.4f}")
            st.metric("Overfitting Score", f"{abs(final_val_loss - final_train_loss):.4f}")

    st.markdown("### Test Cases Results")

    if st.button("Run Test Cases"):
        if st.session_state.embeddings_db is None:
            st.error("Please build a database first in the Person Search page!")
            return

        with st.spinner("Running tests..."):
            try:
                temp_dir = tempfile.mkdtemp()
                image_paths = {}
                for name, data in st.session_state.embeddings_db.items():
                    img_path = os.path.join(temp_dir, name)
                    cv2.imwrite(img_path, cv2.cvtColor(data['image'], cv2.COLOR_RGB2BGR))
                    image_paths[name] = img_path

                test_df = pd.DataFrame({
                    'Anchor': list(image_paths.keys()),
                    'Positive': list(image_paths.keys()),
                    'Negative': list(image_paths.keys())[::-1]
                })

                testset = APN_Dataset(test_df, temp_dir, app.config.INPUT_SIZE, augment=False)
                testloader = DataLoader(testset, batch_size=app.config.BATCH_SIZE, shuffle=False,
                                      num_workers=app.config.NUM_WORKERS, collate_fn=custom_collate)

                tester = PersonReIDTester(app.config)
                sample_img_path = list(image_paths.values())[0]
                final_loss = st.session_state.training_history['val_loss'][-1] if st.session_state.training_history else 0.0

                test_results = tester.run_all_tests(
                    app.model, testloader, temp_dir, test_df, sample_img_path, final_loss)

                cols = st.columns(3)
                for i, (tc, data) in enumerate(test_results.items()):
                    if tc == 'overall':
                        continue
                    with cols[i % 3]:
                        status = "✅ PASS" if data['pass'] else "❌ FAIL"
                        card_class = "success-metric" if data['pass'] else "error-metric"
                        st.markdown(f'''
                        <div class="metric-card {card_class}">
                            <h4>{tc}: {data['description']}</h4>
                            <p>Value: {data['value']:.3f}</p>
                            <p>Threshold: {getattr(app.config, f"{tc.upper()}_THRESHOLD")}</p>
                            <p><strong>{status}</strong></p>
                        </div>
                        ''', unsafe_allow_html=True)

                pass_count = test_results['overall']['pass_count']
                total_tests = test_results['overall']['total']
                st.markdown(f"### Overall Performance: {pass_count}/{total_tests} tests passed ({pass_count/total_tests*100:.1f}%)")

                for img_file in ['tc01_pos_dist.png', 'tc02_neg_dist.png', 'tc03_noise_test.png',
                               'tc04_occlusion_test.png', 'tc05_triplet_example.png', 'tsne_embeddings.png']:
                    if os.path.exists(img_file):
                        st.image(img_file, caption=img_file, use_column_width=True)

            except Exception as e:
                st.error(f"Error running tests: {str(e)}")
                logging.error(f"Test error: {str(e)}")

            finally:
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir, ignore_errors=True)

    if st.session_state.embeddings_db:
        st.markdown("### Embeddings Visualization")
        if st.button("Generate t-SNE Visualization"):
            with st.spinner("Generating t-SNE..."):
                try:
                    embeddings = []
                    labels = []
                    for name, data in st.session_state.embeddings_db.items():
                        embeddings.append(data['embedding'])
                        labels.append(name.split('_')[0] if '_' in name else name[:5])

                    embeddings = np.array(embeddings)
                    if len(embeddings) > 1:
                        perplexity = min(30, len(embeddings) - 1)
                        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                        tsne_results = tsne.fit_transform(embeddings)

                        fig = px.scatter(
                            x=tsne_results[:, 0], 
                            y=tsne_results[:, 1],
                            color=labels,
                            title="t-SNE Visualization of Person Embeddings",
                            labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'}
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough embeddings for t-SNE visualization (minimum 2 required).")
                except Exception as e:
                    st.error(f"Error generating t-SNE: {str(e)}")
                    logging.error(f"t-SNE error: {str(e)}")

def show_settings_page(app):
    """Settings page"""
    st.markdown("## Settings")

    st.markdown("### Model Management")
    uploaded_model = st.file_uploader("Upload trained model (.pt file)", type=['pt'])

    if uploaded_model:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                tmp.write(uploaded_model.getvalue())
                tmp_path = tmp.name

            if app.load_model(tmp_path):
                st.success("Model loaded successfully!")
            os.unlink(tmp_path)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            logging.error(f"Model upload error: {str(e)}")

    st.markdown("### Configuration Parameters")
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            app.config.INPUT_SIZE = st.selectbox(
                "Input Image Size",
                [(128, 128), (224, 224), (256, 256)],
                index=0,
                format_func=lambda x: f"{x[0]}x{x[1]}"
            )
            app.config.NUM_WORKERS = st.slider("DataLoader Workers", 0, 8, app.config.NUM_WORKERS)

        with col2:
            st.markdown("**Test Thresholds**")
            app.config.POS_DIST_THRESHOLD = st.slider("Positive Distance Threshold", 0.1, 1.0, app.config.POS_DIST_THRESHOLD, 0.1)
            app.config.NEG_DIST_THRESHOLD = st.slider("Negative Distance Threshold", 0.5, 2.0, app.config.NEG_DIST_THRESHOLD, 0.1)
            app.config.NOISE_DIST_THRESHOLD = st.slider("Noise Distance Threshold", 0.1, 1.0, app.config.NOISE_DIST_THRESHOLD, 0.1)
            app.config.OCCL_DIST_THRESHOLD = st.slider("Occlusion Distance Threshold", 0.1, 1.0, app.config.OCCL_DIST_THRESHOLD, 0.1)
            app.config.TRIPLET_LOSS_THRESHOLD = st.slider("Triplet Loss Threshold", 0.1, 1.0, app.config.TRIPLET_LOSS_THRESHOLD, 0.1)
            app.config.TOP5_ACCURACY_THRESHOLD = st.slider("Top-5 Accuracy Threshold", 0.5, 1.0, app.config.TOP5_ACCURACY_THRESHOLD, 0.1)

    st.markdown("### System Information")
    st.markdown(f"**Device**: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    st.markdown(f"**PyTorch Version**: {torch.__version__}")
    st.markdown(f"**Current Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()