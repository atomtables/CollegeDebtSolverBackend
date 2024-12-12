# CollegeDebtSolver: hackJPS 2024 Project
## Backend

Adithiya Venkatakrishnan, Alexander Poon, Nathaniel George, Arnav Daine

## Inspiration
Since we are all going to college (probably not), we needed a way to pay for our college. We don't really want to take out huge loans that would basically follow us for the rest of our life. For that, we attempted to create an AI stock predictor.

## What it does
In its completed form, it takes data from Polygon and Yahoo Finance and uses it to train an AI model that would then make predictions that showed if capital gains in the long run would end up being positive or negative, given a time frame of 1 month. However, most of this data was blocked by a paywall, and we did not have a computer with enough RAM to properly and fastly train an AI.

Moreover, paying for a CollegeDebtSolver seems like an oxymoron, and defeats the purpose of this application.

In order to navigate the user interface, you will need to go to the navbar menu and select Dashboard. You can also search for a specific stock by accessing the search menu on the navbar, and searching for that stock. The website is best made for desktop web browsers, such as Google Chrome, Mozilla Firefox, Internet Explorer 6.1, Safari 2.3, and Netscape Navigator.

## How we built it
We discussed ideas before coming into here. We worked our butts off for the past one and a half days to bring it to this level. We used SvelteKit and TailwindCSS for the frontend, and also used DaisyUI for some good-looking completed components.

For the backend, we used FastAPI in order to transfer data to the client. This way, we are able to transfer metadata like icons, stock history, a graph, and statistics of a stock. The FastAPI would have also included AI results that would explain possible predictions for the future.

## Challenges we ran into
Most of the data we needed was hidden behind a paywall, making it inaccessible for most. Also, our computers did not have enough RAM to fastly train an AI model in a way that would make our application fully functional. The AI model we used also had a lot of issues that prevented data from properly being processed. You can see our attempt in our replit backend link.

## Accomplishments that we're proud of
The UI looks phenomenal, and we believe that we put a lot of hard work into making the user interface look good. We also used our resources to the best of our ability, like the paywalled data, and attempted to make the best out of our lemons.

## What we learned
Over the journey, we learned about the simple process of creating a TensorFlow model. Given a couple more days, we would most likely be able to complete the application and AI training, providing a complete interface for what this application would look like.

## What's next for CollegeDebtSolver
We plan on finishing the AI model in order to fully expand the features of this program. The best future for this program is a future in which the program is able to quickly run on sophisticated hardware.

We would like to give credit to Greg Hog for help implementing TensorFlow, and ChatGPT for general debugging of broken code.

### Built With
css
html
javascript
learning
machine
numpy
pandas
svelte
sveltekit
tailwindcss
tensorflow
