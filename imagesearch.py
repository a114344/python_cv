from __future__ import print_function
import matplotlib.pyplot as plt
import nump as np
import pickle
from PIL import Image
from pysqlite2 import dbapi2 as sqlite


class Indexer(object):

    def __init__(self, db, voc):
        """Initialize with the name of the db and a vocabulary object
        """
        self.con = sqlite.connect(db)
        self.voc = voc

    def __del__(self):
        self.con.close()

    def db_commit(self):
        self.con.commit()

    def create_tables(self):
        self.con.execute('CREATE TABLE imlist(filename)')
        self.con.execute('CREATE TABLE imwords(imid, wordid, vocname)')
        self.con.execute('CREATE TABLE imhistograms(imid, wordid, vocname)')
        self.con.execute('CREATE INDEX im_idx ON imlist(filename)')
        self.con.execute('CREATE INDEX wordid_idx ON imwords(wordid)')
        self.con.execute('CREATE INDEX imid_idx ON imwords(imid)')
        self.con.execute('CREATE INDEX imidhist_idx ON imhistograms(imid)')
        self.db.commit()

    def add_to_index(self, imname, descr):
        """Take an image with feature descriptors,

           project on vocabulary, and add to database.
        """
        if self.is_indexed(imname):
            return
        print('Indexing ', + imname)

        # get imid
        imid = self.get_id(imname)

        # get the words
        imwords = self.voc.project(descr)
        nbr_words = imwords.shape[0]

        # link words to images
        for i in range(nbr_words):
            words = imwords[i]
            self.con.execute(
                "INSERT INTO imwords(imid, wordid, vocname) VALUES (?, ?, ?)",
                (imid, wordid, self.voc.name))

        # store word histogram for image
        # use pickle to encode numpy arrays as strings
        self.con.execute("""INSERT INTO imhistograms(imid, histogram, vocname)
                            VALUES (?, ?, ?)""",
                         (imid, pickle.dumps(imwords, self.voc.name)))

    def is_indexed(self, imname):
        """Returns True if imname has been indexed
        """
        im = self.con.execute("""SELECT rowid
                                 FROM imlist
                                 WHERE filename='%s'""" % imname).fetchone()

        return im is not None


def get_id(self, imname):
    """Get an entry id and add if not present
    """

    cur = self.con.execute(
        """SELECT rowid from imlist
           WHERE filename='%s'""" % imname)
    res = cur.fetchone()
    if res is None:
        cur = self.con.execute("""INSERT INTO imlist(filename)
                                  VALUES ('%s')""" % imname)
        return cur.lastrowid
    else:
        return res[0]


def compute_ukbench_score(src, imlist):
    """Returns the average number of correct

       images on the top four results of queries.
    """
    nbr_images = len(imlist)
    pos = np.zeros((nbr_images, 4))
    # get first four results for each image
    for i in range(nbr_images):
        pos[i] = [w[1] - 1 for w in src.query(imlist[i])[:4]]

    # compute score and return average
    score = np.array([(pos[i] // 4) == (i // 4)
                     for i in range(nbr_images)]) * 1.0

    return np.sum(score) / nbr_images


def plot_results(src, res):
    """Show images in result list 'res'.
    """
    plt.figure()
    nbr_results = len(res)
    for i in range(nbr_results):
        imname = src.get_filename(res[i])
        plt.subplot(1, nbr_results, i + 1)
        plt.imshow(np.array(Image.open(imname)))
        plt.axis('off')
    plt.show

    return True


class Searcher(object):

    def __init__(self, db, voc):
        """Initialize the name of the db.
        """

        self.con = sqlite.connect(db)
        self.voc = voc

    def __del__(self):
        self.con.close

    def candidates_from_words(self, imword):
        """Get a list of images containing imword.
        """

        im_ids = self.con.execute("""SELECT DISTINCT imid from imwords
                                     WHERE wordid=%d""" % imword).fetchall()

        return[i[0] for i in im_ids]

    def candidates_from_histogram(self, imwords):
        """Get list of images with similar words.
        """

        # get the word ids
        words = imwords.nonzero()[0]

        # find candidates
        candidates = [self.candidates_from_words(word) for word in words]

        # take all unique words and reverse sort on occurrence
        tmp = [(w, candidates.count(w)) for w in set(candidates)]
        tmp.sort(cmp=lambda x, y: cmp(x[1], y[1]), reverse=True)

        return [w[0] for w in tmp]

    def get_imhistogram(self, imname):
        """Get the word histogram for an image.
        """

        im_id = self.con.execute("""
                                SELECT rowid
                                FROM imlist
                                WHERE filename = '%s' """ % imname).fetchone()

        s = self.con.execute("""
                            SELECT histogram
                            FROM imhistograms
                            WHERE rowid = '%d' """ % im_id).fetchone()

        # use pickle to decode numpy arrays from strings
        return pickle.loads(str(s[0]))

    def query(self, imname):
        """Find a list of matching images for imname
        """
        h = self.candidates_from_histogram()

        matchscores = []
        for imid in candidates:
            # get name
            cand_name = self.con.execute("""
                                        SELECT filename
                                        FROM imlist
                                        WHERE rowid = '%d'""" % imid).fetchone()

            cand_h = self.get_imhistogram(cand_name)
            cand_dist = np.sqrt(np.sum((h - cand_h) ** 2))
            matchscores.append((cand_dist, imid))

            return matchscores.sort()

    def get_filename(self, imid):
        """Return filename for an image id.
        """
        s = self.con.execute(
                            """SELECT filename
                               FROM imlist
                               WHERE rowid = '%d' """ % imid).fetchone()
        return s[0]
