import streamlit as st
from fastai.learner import load_learner
from fastai.vision.core import PILImage
from matplotlib import pyplot as plt

# Title
st.title("Hayvonlarni klassifikatsiya qiluvchi model")

# Rasm yuklash
file = st.file_uploader("Rasm yuklash", type=['png', 'jpg', 'jpeg'])

if file:
    st.image(file, caption="Yuklangan rasm")

    img_fastai = PILImage.create(file)

    # Load model
    model = load_learner("animal_model.pkl")

    # Predict
    pred, pred_id, probs = model.predict(img_fastai)

    # Show result
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id] * 100:.1f}%")

    # plotting using matplotlib
    fig, ax = plt.subplots()

    # Create a bar chart
    ax.bar(model.dls.vocab, probs * 100)  # Convert to percentage
    ax.set_xlabel('Class')
    ax.set_ylabel('Probability (%)')
    ax.set_title('Class Probabilities')

    # Show plot in Streamlit
    st.pyplot(fig)