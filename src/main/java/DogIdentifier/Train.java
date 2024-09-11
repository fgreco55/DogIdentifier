package DogIdentifier;

import javax.imageio.ImageIO;
import javax.visrec.ml.model.ModelCreationException;
import javax.visrec.ml.classification.ImageClassifier;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Enumeration;
import java.util.Map;
import java.util.Properties;

public class Train {
    private static String configFile = "training_data/id.properties";

    public static void main(String[] args) throws ModelCreationException, IOException {

        Properties prop = getConfigProperties(configFile);
        String imagesDir = prop.getProperty("id.imagesDir", "training_data/DogImages2");
        String labelsFile = imagesDir + "/" + prop.getProperty("id.labels", "labels.txt");
        String datasetFile = imagesDir + "/" + prop.getProperty("id.dataset", "small_dataset.txt");
        String archFile = imagesDir + "/" + prop.getProperty("id.arch", "arch.json");
        String modelFile = imagesDir + "/" + prop.getProperty("id.modelFile", "dog-model.dnet");

        System.err.println(
                "ImagesDir: " + imagesDir +  "\n"    +
                "LabelsFile: " + labelsFile + "\n"   +      // specific classification-names to identify
                "DatasetFile: " + datasetFile + "\n" +      // file of pathname to classification-name
                "ArchFile: " + archFile +  "\n"      +      // neural network architecture - default handles most cases
                "ModelFile: " + modelFile);                 // filename of generated model

        // Height and Width should be powers of 2 for best performance - zs
        ImageRecognizer imc = new ImageRecognizer(labelsFile, // specific classification-names to identify
                datasetFile,                                  // file of pathname to classification-name
                archFile,                                     // neural network architecture - default handles most cases
                modelFile,                                    // filename of generated model
                64, 64, 0.02f, 100, 0.001f);

        StopWatch sw = new StopWatch();
        sw.start();
        ImageClassifier<BufferedImage> cl = imc.train();        // number of labels, smaller for faster operation - zs
        sw.stop();

        System.out.println("Training completed in " + sw.getElapsedTimeSecs() + " secs ====================================");

        String fname = imagesDir + "/" + "polly.jpg";
        BufferedImage image = ImageIO.read(new File(fname));   // Polly
        System.out.println("Test using image [" + fname + "]");

        Map<String, Float> results = cl.classify(image);

        System.out.println(results);

    }
    /*
     *  getConfigProperties() - load properties file
     */
    private static Properties getConfigProperties(String fname) throws IOException {
        Properties prop = new Properties();
        InputStream in = new FileInputStream(fname);

        prop.load(in);

        for (Enumeration e = prop.propertyNames(); e.hasMoreElements();) {
            String key = e.nextElement().toString();
            System.out.println(key + " = " + prop.getProperty(key));
        }
        return prop;
    }
}
