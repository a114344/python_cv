from __future__ import print_function
import pickle
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
