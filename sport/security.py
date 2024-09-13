import pandas as pd


class Security:
    """
    An abstraction of single security, single index etc. that carries an ID and a list of attributes
    """

    def __init__(self, sec_id, attributes):
        """
        Args:
            sec_id (str): ID of this Security instance.
            attributes (dict): a dictionary of (attr_name: attr_value).
        """
        if sec_id is None:
            raise ValueError('Input argument sec_id is None')
        if attributes is None:
            raise ValueError('Input argument attributes is None')
        if len(attributes) == 0:
            raise ValueError('Input argument attributes is empty')

        self._sec_id = sec_id
        self._attributes = dict()

        # remove None values
        for attr_name in attributes:
            if attributes.get(attr_name) is not None:
                self._attributes[attr_name.upper()] = attributes.get(attr_name)

    def equals(self, sec2):
        """Compare two securities: True iff attribute names and values match.

        Args:
            sec2 (Security): a Security object.

        Returns:
            result (bool): True if equal and False otherwise.
        """
        if self.get_attr_names() != sec2.get_attr_names():
            return False
        for attr_nm in self.get_attr_names():
            if self.get_attr_value(attr_nm) != sec2.get_attr_value(attr_nm):
                return False

        return True

    def get_attr_names(self):
        """Get the names of all the attributes in this Security instance.

        Returns:
            attr_names (set): a set of attribute names.
        """
        return set(self._attributes.keys())

    def get_attr_value(self, attr_name):
        """ Get the value of an attribute according to its name; None if attribute does not exist.

        Args:
            attr_name (str): name of an attribute.

        Returns:
            value (object): value of an attribute according to its name.
        """
        if attr_name.upper() in self._attributes:
            value = self._attributes.get(attr_name.upper())
        else:
            value = None

        return value

    def get_id(self):
        """Get the ID of this Security instance.

        Returns:
            sec_id (str): id of this Security instance.
        """
        return self._sec_id

    def reproduce(self, new_id=None):
        """Clone a copy of this Security object.

        Args:
            new_id (str): new security ID; use the original if None.

        Returns:
            sec (Security): a clone copy of this Security object.
        """
        sec_id = self.get_id() if new_id is None else new_id
        attributes = {attr_nm: self.get_attr_value(attr_nm) for attr_nm in self.get_attr_names()}

        return Security(sec_id, attributes)

    def reproduce_by_attr(self, attr_nm, attr_value, new_id=None):
        """Reproduce a Security object by updating its attributes; if attribute does not exist, then insert.

        Args:
            attr_nm (str): attribute name.
            attr_value (object): attribute value.
            new_id (str): nw security ID; default to None.

        Returns:
            sec (Security): a new Security object with updated attribute value.
        """
        sec_id = self.get_id() if new_id is None else new_id
        new_attributes = self._attributes.copy()
        new_attributes[attr_nm.upper()] = attr_value

        return Security(sec_id, new_attributes)

    def to_dataframe(self):
        """Convert to data frame representation.

        Returns:
            df (pandas.DataFrame): data frame representation of this Security object.
        """
        data_dict = {'ID': [self.get_id()]}
        for attr_nm in self.get_attr_names():
            data_dict[attr_nm] = [self.get_attr_value(attr_nm)]

        return pd.DataFrame.from_dict(data_dict).set_index(keys=['ID'], drop=True)